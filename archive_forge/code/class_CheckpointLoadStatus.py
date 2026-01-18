import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class CheckpointLoadStatus(_LoadStatus):
    """Checks the status of checkpoint loading and manages restore ops.

  Returned from `Saver.restore`. Since `restore` may defer the loading of values
  in the checkpoint which don't yet have corresponding Python objects,
  `CheckpointLoadStatus` provides a callback to verify that checkpoint loading
  is complete (`assert_consumed`).

  When graph building, `restore` does not run restore ops itself since their
  creation may be deferred. The `run_restore_ops` method must be called once all
  Python objects with values to restore have been created and added to the
  dependency graph (this does not necessarily have to be the whole checkpoint;
  calling `run_restore_ops` while `assert_consumed` fails is supported and will
  partially restore the checkpoint).

  See `Saver.restore` for usage examples.
  """

    def __init__(self, checkpoint, feed_dict, graph_view):
        self._checkpoint = checkpoint
        self._feed_dict = feed_dict
        self._object_graph_view = graph_view
        self._root = graph_view.root

    def assert_consumed(self):
        """Asserts that all objects in the checkpoint have been created/matched.

    Returns:
      `self` for chaining.
    Raises:
      AssertionError: If there are any Python objects in the dependency graph
        which have not been restored from this checkpoint or a later `restore`,
        or if there are any checkpointed values which have not been matched to
        Python objects.
    """
        pretty_printer = ObjectGraphProtoPrettyPrinter(self._checkpoint.object_graph_proto)
        self.assert_existing_objects_matched()
        for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
            if not node.attributes:
                continue
            trackable = self._checkpoint.object_by_proto_id.get(node_id, None)
            if trackable is None:
                raise AssertionError(f'Unresolved object in checkpoint {pretty_printer.node_names[node_id]}: {node}')
        if self._checkpoint.slot_restorations:
            raise AssertionError(f'Unresolved slot restorations: {self._checkpoint.slot_restorations}')
        if self._checkpoint.unused_attributes:
            unused_attribute_messages = []
            for node_id, attribute in self._checkpoint.unused_attributes.items():
                obj = self._checkpoint.object_by_proto_id[node_id]
                unused_attribute_messages.append(f'{pretty_printer.node_names[node_id]} ({obj}): {attribute}')
            joined_attribute_messages = '\n'.join(unused_attribute_messages)
            raise AssertionError(f'Unused attributes in these objects (the attributes exist in the checkpoint but were not restored):\n{joined_attribute_messages}')
        return self

    def assert_existing_objects_matched(self):
        """Asserts that trackable Python objects have been matched.

    Note that this is a weaker assertion than `assert_consumed`. It will only
    fail for existing Python objects which are (transitive) dependencies of the
    root object and which do not have an entry in the checkpoint.

    It will not fail, for example, if a `tf.keras.Layer` object has not yet been
    built and so has not created any `tf.Variable` objects.

    Returns:
      `self` for chaining.

    Raises:
      AssertionError: If a Python object exists in the transitive dependencies
        of the root object but does not have a value in the checkpoint.
    """
        for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
            trackable = self._checkpoint.object_by_proto_id.get(node_id, None)
            if trackable is not None and trackable._update_uid < self._checkpoint.restore_uid:
                raise AssertionError(f'Object {node} not assigned a value from checkpoint.')
        for trackable_object in util.list_objects(self._object_graph_view):
            if isinstance(trackable_object, data_structures.TrackableDataStructure) and (not trackable_object._trackable_children(save_type=base.SaveType.CHECKPOINT)):
                continue
            self._checkpoint.all_python_objects.add(trackable_object)
        unused_python_objects = object_identity.ObjectIdentitySet(_objects_with_attributes(self._checkpoint.all_python_objects)) - object_identity.ObjectIdentitySet(self._checkpoint.object_by_proto_id.values())
        if unused_python_objects:
            num_unused_python_objects = len(list(unused_python_objects))
            num_variables_to_show = min(10, num_unused_python_objects)
            raise AssertionError(f'Found {num_unused_python_objects} Python objects that were not bound to checkpointed values, likely due to changes in the Python program. Showing {num_variables_to_show} of {num_unused_python_objects} unmatched objects: {list(unused_python_objects)[:num_variables_to_show]}')
        return self

    def assert_nontrivial_match(self):
        """Raises an exception if only the root object matched."""
        for trackable_object in util.list_objects(self._object_graph_view):
            self._checkpoint.all_python_objects.add(trackable_object)
        if len(self._checkpoint.object_by_proto_id) <= 1:
            unused_python_objects = object_identity.ObjectIdentitySet(_objects_with_attributes(self._checkpoint.all_python_objects)) - object_identity.ObjectIdentitySet(self._checkpoint.object_by_proto_id.values())
            if unused_python_objects:
                raise AssertionError(f'Nothing except the root object matched a checkpointed value. Typically this means that the checkpoint does not match the Python program. The following objects have no matching checkpointed value: {list(unused_python_objects)}')
            else:
                raise AssertionError(f'Nothing to load. No dependencies have been added to {self._object_graph_view.root} yet.')
        return self

    def run_restore_ops(self, session=None):
        """Run operations to restore objects in the dependency graph."""
        if context.executing_eagerly():
            return
        if session is None:
            session = get_session()
        session.run(self._checkpoint.restore_ops, feed_dict=self._feed_dict)

    def initialize_or_restore(self, session=None):
        """Run operations to initialize or restore objects in the dependency graph.

    Any objects in the dependency graph which have initializers but are not in
    the checkpoint will have those initializers run, unless those variables are
    being restored by a later call to `tf.train.Checkpoint.restore()`.

    This method has a sibling in `InitializationOnlyStatus` which instead
    initializes variables. That type is returned if no checkpoint is specified
    in `Saver.restore`.

    Args:
      session: The session to run init/restore ops in. If `None`, uses the
        default session.
    """
        if context.executing_eagerly():
            return
        if session is None:
            session = get_session()
        all_objects = util.list_objects(self._object_graph_view)
        already_initialized_objects = object_identity.ObjectIdentitySet(self._checkpoint.object_by_proto_id.values())
        initializers_for_non_restored_variables = [c.initializer for c in all_objects if hasattr(c, 'initializer') and c not in already_initialized_objects and (getattr(c, '_update_uid', self._checkpoint.restore_uid - 1) < self._checkpoint.restore_uid)]
        self.run_restore_ops(session=session)
        session.run(initializers_for_non_restored_variables)

    def expect_partial(self):
        """Silence warnings about incomplete checkpoint restores."""
        self._checkpoint.expect_partial = True
        return self
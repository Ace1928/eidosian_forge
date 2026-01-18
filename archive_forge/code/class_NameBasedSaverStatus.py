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
class NameBasedSaverStatus(_LoadStatus):
    """Status for loading a name-based training checkpoint."""

    @deprecation.deprecated(date=None, instructions=_DEPRECATED_RESTORE_INSTRUCTIONS)
    def __init__(self, checkpoint, object_graph_view):
        self._checkpoint = checkpoint
        self._object_graph_view = object_graph_view
        self._optionally_restored = []
        self._root = object_graph_view.root

    def add_to_optionally_restored(self, var):
        """Add a variable to the list of optionally restored variables.

    There are situations where certain variables should be ignored in assertions
    such as assert_existing_objects_matched(). One example is that of a
    checkpoint saved with train.Saver(), and restored with train.Checkpoint():
    it is possible for the train.Saver() checkpoint to be missing the internal
    `save_counter` variable, which we want to ignore on restore.

    Args:
      var: The variable to treat as optionally restored.
    """
        self._optionally_restored.append(var)

    def assert_consumed(self):
        """Raises an exception if any variables are unmatched."""
        unused_attributes = list(self._checkpoint.unused_attributes.items())
        unused_attributes = [a for a in unused_attributes if all((a[0] is not x for x in self._optionally_restored))]
        if unused_attributes:
            unused_attribute_string = ''.join((f'\n    {obj}: {attributes}' for obj, attributes in unused_attributes))
            raise AssertionError(f'Some objects had attributes which were not restored: {unused_attribute_string}')
        for trackable in util.list_objects(self._object_graph_view):
            trackable._maybe_initialize_trackable()
            if trackable._update_uid < self._checkpoint.restore_uid:
                raise AssertionError(f'Object not restored: {trackable}')
        return self

    def assert_existing_objects_matched(self):
        """Raises an exception if currently created objects are unmatched."""
        return self.assert_consumed()

    def assert_nontrivial_match(self):
        """Raises an exception if currently created objects are unmatched."""
        return self.assert_consumed()

    def _gather_saveable_objects(self):
        """Walk the object graph, using global names for SaveableObjects."""
        objects = util.list_objects(self._object_graph_view)
        saveable_objects = []
        for trackable in objects:
            trackable._maybe_initialize_trackable()
            if trackable._update_uid < self._checkpoint.restore_uid:
                trackable._update_uid = self._checkpoint.restore_uid
            else:
                continue
            saveable_objects.extend(self._checkpoint.globally_named_object_attributes(trackable))
        return saveable_objects

    def run_restore_ops(self, session=None):
        """Load the name-based checkpoint using a new `tf.compat.v1.train.Saver`."""
        if context.executing_eagerly():
            return
        if session is None:
            session = get_session()
        with ops.device('/cpu:0'):
            saveables = self._gather_saveable_objects()
            v1_saver_lib.Saver(saveables).restore(sess=session, save_path=self._checkpoint.save_path)

    def initialize_or_restore(self, session=None):
        """Alias for `run_restore_ops`."""
        self.run_restore_ops(session=session)
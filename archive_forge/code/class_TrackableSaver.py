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
class TrackableSaver:
    """Saves and restores a `Trackable` object and its dependencies.

  See `Trackable` for details of dependency management. `Saver` wraps
  `tf.compat.v1.train.Saver` for saving, including extra information about the
  graph of
  dependencies between Python objects. When restoring, it uses this information
  about the save-time dependency graph to more robustly match objects with their
  checkpointed values. When executing eagerly, it supports restoring variables
  on object creation (see `Saver.restore`).

  Values in a checkpoint are mapped to `Trackable` Python objects
  (`Variable`s, `Optimizer`s, `Layer`s) based on the names provided when the
  checkpoint was written. To avoid breaking existing checkpoints when modifying
  a class, dependency names (the names of attributes to which `Trackable`
  objects are assigned) may not change. These names are local to objects, in
  contrast to the `Variable.name`-based save/restore from
  `tf.compat.v1.train.Saver`, and
  so allow additional program transformations.
  """

    def __init__(self, graph_view):
        """Configure saving.

    Args:
      graph_view: An `ObjectGraphView` object containing a description of the
        object graph to save.
    """
        self._graph_view = graph_view
        if context.executing_eagerly():
            self._cache = None
            self._saveables_cache = None
        else:
            self._cache = object_identity.ObjectIdentityWeakKeyDictionary()
            self._saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()
        self._file_prefix_placeholder = None
        self._object_graph_feed_tensor = None
        self._last_save_object_graph = None
        self._file_prefix_feed_tensor = None
        self._cached_save_operation = None
        self._restore_op_cache = {}
        self._object_map = None

    def _gather_serialized_tensors(self, object_graph_tensor=None):
        """Gathers tensors to save to ckpt and includes the object graph proto."""
        serialized_tensors, feed_additions, registered_savers, graph_proto = save_util.serialize_graph_view(self._graph_view, self._object_map, cache=self._cache)
        if self._saveables_cache is not None:
            self._saveables_cache = saveable_object_util.serialized_tensors_to_saveable_cache(serialized_tensors)
        if object_graph_tensor is None:
            with ops.device('/cpu:0'):
                object_graph_tensor = constant_op.constant(graph_proto.SerializeToString(), dtype=dtypes.string)
        else:
            feed_additions.update({object_graph_tensor: graph_proto.SerializeToString()})
        assert base.OBJECT_GRAPH_PROTO_KEY not in serialized_tensors.get(None, {})
        serialized_tensors.setdefault(None, {})[base.OBJECT_GRAPH_PROTO_KEY] = object_graph_tensor
        return (serialized_tensors, feed_additions, registered_savers, graph_proto)

    def _save_cached_when_graph_building(self, file_prefix, object_graph_tensor, options):
        """Create or retrieve save ops.

    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.

    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
        serialized_tensors, feed_additions, registered_savers, graph_proto = self._gather_serialized_tensors(object_graph_tensor)
        if self._last_save_object_graph != graph_proto or context.executing_eagerly() or ops.inside_function():
            saver = functional_saver.MultiDeviceSaver(serialized_tensors, registered_savers)
            save_op = saver.save(file_prefix, options=options)
            with ops.device('/cpu:0'):
                with ops.control_dependencies([save_op]):
                    self._cached_save_operation = array_ops.identity(file_prefix)
            self._last_save_object_graph = graph_proto
        return (self._cached_save_operation, feed_additions)

    def save(self, file_prefix, checkpoint_number=None, session=None, options=None):
        """Save a training checkpoint.

    The saved checkpoint includes variables created by this object and any
    Trackable objects it depends on at the time `Saver.save()` is called.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `checkpoint_number`, if provided.
      checkpoint_number: An integer variable or Tensor, used to number
        checkpoints. Typically this value is saved along with other variables in
        training checkpoints, which will happen automatically if it was created
        by `root_trackable` or one of its dependencies (via
        `Trackable._add_variable`).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      The full path to the checkpoint.

    Raises:
      RuntimeError: if called in V1 Graph mode without a default session.
    """
        options = options or checkpoint_options.CheckpointOptions()
        feed_dict = {}
        use_session = not context.executing_eagerly() and (not ops.inside_function())
        if checkpoint_number:
            file_prefix = '%s-%d' % (file_prefix, checkpoint_number)
        if use_session:
            if self._object_graph_feed_tensor is None:
                with ops.device('/cpu:0'):
                    self._object_graph_feed_tensor = constant_op.constant('', dtype=dtypes.string)
                    self._file_prefix_feed_tensor = constant_op.constant('', dtype=dtypes.string)
            object_graph_tensor = self._object_graph_feed_tensor
            file_prefix_tensor = self._file_prefix_feed_tensor
            feed_dict[file_prefix_tensor] = file_prefix
        else:
            with ops.device('/cpu:0'):
                file_prefix_tensor = ops.convert_to_tensor(file_prefix, dtype=dtypes.string)
            object_graph_tensor = None
        if not tensor_util.is_tensor(file_prefix):
            file_io.recursive_create_dir(os.path.dirname(file_prefix))
        save_path, new_feed_additions = self._save_cached_when_graph_building(file_prefix_tensor, object_graph_tensor, options)
        if new_feed_additions:
            feed_dict.update(new_feed_additions)
        if not use_session:
            session = None
        elif session is None:
            session = get_session()
        if session:
            return session.run(save_path, feed_dict=feed_dict)
        elif use_session:
            raise RuntimeError(f'Unable to save checkpoint to "{file_prefix}" in graph mode without a default session. Please use `with tf.Session():` to create a session.')
        else:
            return save_path

    def restore(self, save_path, options=None):
        """Restore a training checkpoint.

    Restores `root_trackable` and any objects that it tracks
    (transitive). Either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added to the `root_trackable` passed to the
    constructor after this call will be matched if they have a corresponding
    object in the checkpoint.

    When building a graph, restorations are added to the graph but not run.

    ```python
    saver = Saver(root)
    saver.restore(path)
    ```

    To ensure that loading is complete and no more deferred restorations will
    take place, you can use the `assert_consumed()` method of the status object
    returned by the `restore` call.

    The assert will raise an exception unless every object was matched and all
    checkpointed values have a matching variable object.

    ```python
    saver = Saver(root)
    saver.restore(path).assert_consumed()
    ```

    When graph building, `assert_consumed()` indicates that all of the restore
    ops which will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` function of the status object:

    ```python
    saver.restore(path).assert_consumed().run_restore_ops()
    ```

    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.

    Name-based `tf.compat.v1.train.Saver` checkpoints can be loaded using this
    method. There is no deferred loading, and names are used to match
    variables. No restore ops are created/run until `run_restore_ops()` or
    `initialize_or_restore()` are called on the returned status object, even
    when executing eagerly. Re-encode name-based checkpoints using this
    object-based `Saver.save` as soon as possible.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      A load status object, which can be used to make assertions about the
      status of checkpoint restoration and run initialization/restore ops
      (of type `CheckpointLoadStatus`, or `InitializationOnlyStatus` if
      `save_path` is `None`).

      If `save_path` points to a name-based checkpoint, a `NameBasedSaverStatus`
      object is returned which runs restore ops from a name-based saver.

    Raises:
      RuntimeError: When a checkpoint file saved by async checkpoint is not
        available upon restore().
    """
        options = options or checkpoint_options.CheckpointOptions()
        if save_path is None:
            return InitializationOnlyStatus(self._graph_view, ops.uid())
        global _ASYNC_CHECKPOINT_THREAD
        if _ASYNC_CHECKPOINT_THREAD is not None:
            _ASYNC_CHECKPOINT_THREAD.join()
        reader = py_checkpoint_reader.NewCheckpointReader(save_path)
        graph_building = not context.executing_eagerly()
        if graph_building:
            dtype_map = None
        else:
            dtype_map = reader.get_variable_to_dtype_map()
        try:
            object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
        except errors_impl.NotFoundError:
            restore_coordinator = _NameBasedRestoreCoordinator(save_path=save_path, dtype_map=dtype_map)
            if not graph_building:
                for existing_trackable in util.list_objects(self._graph_view):
                    existing_trackable._maybe_initialize_trackable()
                    existing_trackable._name_based_restores.add(restore_coordinator)
                    existing_trackable._name_based_attribute_restore(restore_coordinator)
            return NameBasedSaverStatus(restore_coordinator, object_graph_view=self._graph_view)
        if graph_building:
            if self._file_prefix_placeholder is None:
                with ops.device('/cpu:0'):
                    self._file_prefix_placeholder = constant_op.constant('model')
            file_prefix_tensor = self._file_prefix_placeholder
            file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
        else:
            with ops.device('/cpu:0'):
                file_prefix_tensor = constant_op.constant(save_path)
            file_prefix_feed_dict = None
        object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
        object_graph_proto.ParseFromString(object_graph_string)
        checkpoint = _CheckpointRestoreCoordinator(object_graph_proto=object_graph_proto, save_path=save_path, save_path_tensor=file_prefix_tensor, reader=reader, restore_op_cache=self._restore_op_cache, graph_view=self._graph_view, options=options, saveables_cache=self._saveables_cache)
        restore_lib.CheckpointPosition(checkpoint=checkpoint, proto_id=0).restore(self._graph_view.root, reader)
        if self._graph_view.attached_dependencies:
            for ref in self._graph_view.attached_dependencies:
                if ref.name == 'root':
                    continue
                proto_id = None
                for proto_ref in object_graph_proto.nodes[0].children:
                    if proto_ref.local_name == ref.name:
                        proto_id = proto_ref.node_id
                        break
                if proto_id in checkpoint.object_by_proto_id:
                    continue
                if proto_id is None:
                    continue
                restore_lib.CheckpointPosition(checkpoint=checkpoint, proto_id=proto_id).restore(ref.ref, reader)
        load_status = CheckpointLoadStatus(checkpoint, graph_view=self._graph_view, feed_dict=file_prefix_feed_dict)
        return load_status
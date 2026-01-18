from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class DTrackableSaver(util.TrackableSaver):
    """A DTensor trackable saver that uses _SingleDeviceSaver."""

    def __init__(self, mesh: layout.Mesh, graph_view):
        super(DTrackableSaver, self).__init__(graph_view)
        self._mesh = mesh

    def _gather_saveables(self, object_graph_tensor=None):
        serialized_tensors, feed_additions, registered_savers, graph_proto = self._gather_serialized_tensors(object_graph_tensor)
        saveables_dict = self._saveables_cache
        if saveables_dict is None:
            object_graph_tensor = serialized_tensors.pop(None)[base.OBJECT_GRAPH_PROTO_KEY]
            saveables_dict = saveable_object_util.serialized_tensors_to_saveable_cache(serialized_tensors)
        named_saveable_objects = []
        for saveable_by_name in saveables_dict.values():
            for saveables in saveable_by_name.values():
                named_saveable_objects.extend(saveables)
        named_saveable_objects.append(base.NoRestoreSaveable(tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
        return (named_saveable_objects, graph_proto, feed_additions, registered_savers)

    def _save_cached_when_graph_building(self, file_prefix, object_graph_tensor, options, update_ckpt_state=False):
        """Create or retrieve save ops, overrides parents's private method.

    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated. This is used for async checkpoint,
        which DTrackableSaver currently does not support.
    TODO(chienchunh): Implement async checkpoint for DTrackableSaver.

    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
        named_saveable_objects, graph_proto, feed_additions, unused_registered_savers = self._gather_saveables(object_graph_tensor=object_graph_tensor)
        if self._last_save_object_graph != graph_proto or context.executing_eagerly() or ops.inside_function():
            saver = _DSaver(self._mesh, named_saveable_objects)
            save_op = saver.save(file_prefix, options=options)
            with ops.device('/cpu:0'):
                with ops.control_dependencies([save_op]):
                    self._cached_save_operation = array_ops.identity(file_prefix)
            self._last_save_object_graph = graph_proto
        return (self._cached_save_operation, feed_additions)

    def restore(self, save_path, options=None):
        """Restore a training checkpoint with host mesh placement."""
        options = options or checkpoint_options.CheckpointOptions()
        if save_path is None:
            return util.InitializationOnlyStatus(self._graph_view, ops.uid())
        reader = py_checkpoint_reader.NewCheckpointReader(save_path)
        graph_building = not context.executing_eagerly()
        if graph_building:
            dtype_map = None
        else:
            dtype_map = reader.get_variable_to_dtype_map()
        try:
            object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
        except errors_impl.NotFoundError:
            restore_coordinator = util._NameBasedRestoreCoordinator(save_path=save_path, dtype_map=dtype_map)
            if not graph_building:
                for existing_trackable in self._graph_view.list_objects():
                    existing_trackable._maybe_initialize_trackable()
                    existing_trackable._name_based_restores.add(restore_coordinator)
                    existing_trackable._name_based_attribute_restore(restore_coordinator)
            return util.NameBasedSaverStatus(restore_coordinator, graph_view=self._graph_view)
        if graph_building:
            if self._file_prefix_placeholder is None:
                self._file_prefix_placeholder = api.pack([constant_op.constant('model')] * self._mesh.num_local_devices(), layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
            file_prefix_tensor = self._file_prefix_placeholder
            file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
        else:
            file_prefix_tensor = api.pack([constant_op.constant(save_path)] * self._mesh.num_local_devices(), layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
            file_prefix_feed_dict = None
        object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
        object_graph_proto.ParseFromString(object_graph_string)
        checkpoint = _DCheckpointRestoreCoordinator(mesh=self._mesh, object_graph_proto=object_graph_proto, save_path=save_path, save_path_tensor=file_prefix_tensor, reader=reader, restore_op_cache=self._restore_op_cache, graph_view=self._graph_view, options=options, saveables_cache=self._saveables_cache)
        restore_lib.CheckpointPosition(checkpoint=checkpoint, proto_id=0).restore(self._graph_view.root)
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
                restore_lib.CheckpointPosition(checkpoint=checkpoint, proto_id=proto_id).restore(ref.ref)
        load_status = util.CheckpointLoadStatus(checkpoint, graph_view=self._graph_view, feed_dict=file_prefix_feed_dict)
        return load_status
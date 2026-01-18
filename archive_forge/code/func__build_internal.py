import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _build_internal(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, filename='model', build_save=True, build_restore=True):
    """build() with option to only perform save and restore."""
    if not context.executing_eagerly() and (not build_save or not build_restore):
        raise ValueError('save and restore operations need to be built together  when eager execution is not enabled.')
    if not isinstance(names_to_saveables, dict):
        names_to_saveables = saveable_object_util.op_list_to_dict(names_to_saveables)
    saveables = saveable_object_util.validate_and_slice_inputs(names_to_saveables)
    if max_to_keep is None:
        max_to_keep = 0
    with ops.name_scope(name, 'save', [saveable.op for saveable in saveables]) as name:
        filename_tensor = array_ops.placeholder_with_default(filename or 'model', shape=(), name='filename')
        filename_tensor = array_ops.placeholder_with_default(filename_tensor, shape=(), name='Const')
        if sharded:
            per_device = self._GroupByDevices(saveables)
            if build_save:
                save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
            if build_restore:
                restore_op = self._AddShardedRestoreOps(filename_tensor, per_device, restore_sequentially, reshape)
        else:
            if build_save:
                save_tensor = self._AddSaveOps(filename_tensor, saveables)
            if build_restore:
                restore_op = self._AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape)
    if context.executing_eagerly():
        save_tensor_name = save_tensor.numpy() if build_save else ''
        return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.numpy(), save_tensor_name=save_tensor_name, restore_op_name='', max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)
    else:
        graph = ops.get_default_graph()
        check_collection_list = graph.get_all_collection_keys()
        for collection_type in check_collection_list:
            for element in graph.get_collection(collection_type):
                if isinstance(element, variables.PartitionedVariable):
                    try:
                        graph.get_operation_by_name(element.name)
                    except KeyError:
                        element.as_tensor()
        return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.name, save_tensor_name=save_tensor.name, restore_op_name=restore_op.name, max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)
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
def _AddShardedSaveOps(self, filename_tensor, per_device):
    """Add ops to save the params per shard.

    Args:
      filename_tensor: a scalar String Tensor.
      per_device: A list of (device, BaseSaverBuilder.SaveableObject) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables.
    """
    if self._write_version == saver_pb2.SaverDef.V2:
        return self._AddShardedSaveOpsForV2(filename_tensor, per_device)
    num_shards = len(per_device)
    sharded_saves = []
    num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
    for shard, (device, saveables) in enumerate(per_device):
        with ops.device(device):
            sharded_filename = self.sharded_filename(filename_tensor, shard, num_shards_tensor)
            sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
    with ops.control_dependencies([x.op for x in sharded_saves]):
        return gen_io_ops.sharded_filespec(filename_tensor, num_shards_tensor)
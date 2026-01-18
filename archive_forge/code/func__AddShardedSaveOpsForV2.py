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
def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
    """Add ops to save the params per shard, for the V2 format.

    Note that the sharded save procedure for the V2 format is different from
    V1: there is a special "merge" step that merges the small metadata produced
    from each device.

    Args:
      checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A FILENAME*,
        but as a prefix of a V2 checkpoint;
      per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables, which, when evaluated, returns the prefix
        "<user-fed prefix>" only and does not include the sharded spec suffix.
    """
    with ops.device('CPU'):
        _SHARDED_SUFFIX = array_ops.where(string_ops.regex_full_match(checkpoint_prefix, '^s3://.*'), constant_op.constant('.part'), constant_op.constant(os.path.normpath('_temp/part')))
        tmp_checkpoint_prefix = string_ops.string_join([checkpoint_prefix, _SHARDED_SUFFIX])
    num_shards = len(per_device)
    sharded_saves = []
    sharded_prefixes = []
    num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
    last_device = None
    for shard, (device, saveables) in enumerate(per_device):
        last_device = device
        with ops.device(saveable_object_util.set_cpu0(device)):
            sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard, num_shards_tensor)
            sharded_prefixes.append(sharded_filename)
            sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
    with ops.control_dependencies([x.op for x in sharded_saves]):
        with ops.device(saveable_object_util.set_cpu0(last_device)):
            merge_step = gen_io_ops.merge_v2_checkpoints(sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
            with ops.control_dependencies([merge_step]):
                return array_ops.identity(checkpoint_prefix)
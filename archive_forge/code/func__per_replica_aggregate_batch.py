import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _per_replica_aggregate_batch(strategy, batch_outs, model, mode):
    """Aggregates the per-replica batch-level outputs from a distributed step."""
    if strategy is not None and mode == ModeKeys.PREDICT:
        total_batch_outs = []
        for i in range(len(model.outputs)):
            num_replicas = strategy.num_replicas_in_sync
            nested_outs = batch_outs[i * num_replicas:i * num_replicas + num_replicas]
            total_batch_outs.append(concat_along_batch_dimension(nest.flatten(nested_outs)))
        return total_batch_outs
    return batch_outs
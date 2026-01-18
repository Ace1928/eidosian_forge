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
def _make_eager_execution_function(model, mode):
    """Makes function to run one step of distributed model eager execution."""

    def _per_replica_function(model):
        f = model._make_execution_function(mode)
        return (f.inputs, f.outputs)
    strategy = model._distribution_strategy
    global_graph = backend.get_graph()
    with global_graph.as_default(), strategy.scope():
        with backend._scratch_graph(global_graph):
            grouped = strategy.extended.call_for_each_replica(_per_replica_function, args=(get_distributed_model(model, mode),))
            grouped_inputs, grouped_outputs = grouped
            all_inputs, all_outputs, _, _ = unwrap_values(strategy, grouped_inputs, grouped_outputs, with_loss_tensor=mode != ModeKeys.PREDICT)
        return backend.function(all_inputs, all_outputs, name='eager_distributed_{}_function'.format(mode))
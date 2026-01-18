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
def _make_execution_function_without_cloning(model, mode):
    """Creates a function to run one step of distributed model execution."""
    strategy = model._distribution_strategy
    with strategy.scope():
        per_replica_function = _make_replica_execution_function(model, mode)

        def distributed_function(input_fn):
            """A single step of the distributed execution across replicas."""
            x, y, sample_weights = input_fn()
            outputs = strategy.run(per_replica_function, args=(x, y, sample_weights))
            all_outputs = unwrap_outputs(strategy, outputs, with_loss_tensor=mode != ModeKeys.PREDICT)
            return all_outputs
        if not model.run_eagerly:
            distributed_function = def_function.function(distributed_function)

            def execution_function(input_fn):
                return [out.numpy() for out in distributed_function(input_fn)]
        else:
            execution_function = distributed_function
        return execution_function
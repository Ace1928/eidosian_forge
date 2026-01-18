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
def _make_execution_function_with_cloning(model, mode):
    """Clones or re-uses models to run one step of distributed model execution."""
    distributed_model = get_distributed_model(model, mode)
    if distributed_model and hasattr(distributed_model, '_distribution_function') and (not (hasattr(distributed_model, '_recompile_exec_function') and distributed_model._recompile_exec_function)):
        return distributed_model._distributed_function
    if not distributed_model:
        _make_replicated_models_with_cloning(model, mode)
        distributed_model = get_distributed_model(model, mode)
    assert distributed_model
    if context.executing_eagerly():
        distributed_function = _make_eager_execution_function(model, mode)
    else:
        distributed_function = _make_graph_execution_function(model, mode)
    distributed_model._distributed_function = distributed_function
    distributed_model._recompile_exec_function = False
    return distributed_function
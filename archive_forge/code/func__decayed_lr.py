import abc
import contextlib
import functools
import warnings
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _decayed_lr(self, var_dtype):
    """Get decayed learning rate as a Tensor with dtype=var_dtype."""
    lr_t = self._get_hyper('learning_rate', var_dtype)
    if isinstance(lr_t, learning_rate_schedule.LearningRateSchedule):
        local_step = math_ops.cast(self.iterations, var_dtype)
        lr_t = math_ops.cast(lr_t(local_step), var_dtype)
    if self._initial_decay > 0.0:
        local_step = math_ops.cast(self.iterations, var_dtype)
        decay_t = math_ops.cast(self._initial_decay, var_dtype)
        lr_t = lr_t / (1.0 + decay_t * local_step)
    return lr_t
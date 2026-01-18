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
def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: `Tensor` or callable. If a callable, `loss` should take no
        arguments and return the value to minimize. If a `Tensor`, the `tape`
        argument must be passed.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      tape: (Optional) `tf.GradientTape`. If `loss` is provided as a `Tensor`,
        the tape that computed the `loss` must be provided.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid, or var_list is None.
    """
    if not callable(loss) and tape is None:
        raise ValueError('`tape` is required when a `Tensor` loss is passed.')
    tape = tape if tape is not None else backprop.GradientTape()
    if callable(loss):
        with tape:
            if not callable(var_list):
                tape.watch(var_list)
            loss = loss()
            if callable(var_list):
                var_list = var_list()
    with tape:
        loss = self._transform_loss(loss)
    var_list = nest.flatten(var_list)
    with ops.name_scope_v2(self._name + '/gradients'):
        grads_and_vars = self._get_gradients(tape, loss, var_list, grad_loss)
    self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars
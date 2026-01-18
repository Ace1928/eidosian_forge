import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _standardize_args(inputs, initial_state, constants, num_constants):
    """Standardizes `__call__` to a single list of tensor inputs.

  When running a model loaded from a file, the input tensors
  `initial_state` and `constants` can be passed to `RNN.__call__()` as part
  of `inputs` instead of by the dedicated keyword arguments. This method
  makes sure the arguments are separated and that `initial_state` and
  `constants` are lists of tensors (or None).

  Args:
    inputs: Tensor or list/tuple of tensors. which may include constants
      and initial states. In that case `num_constant` must be specified.
    initial_state: Tensor or list of tensors or None, initial states.
    constants: Tensor or list of tensors or None, constant tensors.
    num_constants: Expected number of constants (if constants are passed as
      part of the `inputs` list.

  Returns:
    inputs: Single tensor or tuple of tensors.
    initial_state: List of tensors or None.
    constants: List of tensors or None.
  """
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
            inputs = inputs[:1]
        if len(inputs) > 1:
            inputs = tuple(inputs)
        else:
            inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]
    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)
    return (inputs, initial_state, constants)
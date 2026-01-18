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
@staticmethod
def _validate_state_spec(cell_state_sizes, init_state_specs):
    """Validate the state spec between the initial_state and the state_size.

    Args:
      cell_state_sizes: list, the `state_size` attribute from the cell.
      init_state_specs: list, the `state_spec` from the initial_state that is
        passed in `call()`.

    Raises:
      ValueError: When initial state spec is not compatible with the state size.
    """
    validation_error = ValueError('An `initial_state` was passed that is not compatible with `cell.state_size`. Received `state_spec`={}; however `cell.state_size` is {}'.format(init_state_specs, cell_state_sizes))
    flat_cell_state_sizes = nest.flatten(cell_state_sizes)
    flat_state_specs = nest.flatten(init_state_specs)
    if len(flat_cell_state_sizes) != len(flat_state_specs):
        raise validation_error
    for cell_state_spec, cell_state_size in zip(flat_state_specs, flat_cell_state_sizes):
        if not tensor_shape.TensorShape(cell_state_spec.shape[1:]).is_compatible_with(tensor_shape.TensorShape(cell_state_size)):
            raise validation_error
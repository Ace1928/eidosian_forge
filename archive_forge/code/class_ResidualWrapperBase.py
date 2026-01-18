import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
class ResidualWrapperBase(object):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(self, cell, residual_fn=None, **kwargs):
        """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
          and outputs.
      **kwargs: dict of keyword arguments for base layer.
    """
        super(ResidualWrapperBase, self).__init__(cell, **kwargs)
        self._residual_fn = residual_fn

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope_v2(type(self).__name__ + 'ZeroState'):
            return self.cell.zero_state(batch_size, dtype)

    def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
        """Run the cell and then apply the residual_fn on its inputs to its outputs.

    Args:
      inputs: cell inputs.
      state: cell state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments passed to the wrapped cell's `call`.

    Returns:
      Tuple of cell outputs and new state.

    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
        outputs, new_state = cell_call_fn(inputs, state, **kwargs)

        def assert_shape_match(inp, out):
            inp.get_shape().assert_is_compatible_with(out.get_shape())

        def default_residual_fn(inputs, outputs):
            nest.assert_same_structure(inputs, outputs)
            nest.map_structure(assert_shape_match, inputs, outputs)
            return nest.map_structure(lambda inp, out: inp + out, inputs, outputs)
        res_outputs = (self._residual_fn or default_residual_fn)(inputs, outputs)
        return (res_outputs, new_state)

    def get_config(self):
        """Returns the config of the residual wrapper."""
        if self._residual_fn is not None:
            function, function_type, function_module = _serialize_function_to_config(self._residual_fn)
            config = {'residual_fn': function, 'residual_fn_type': function_type, 'residual_fn_module': function_module}
        else:
            config = {}
        base_config = super(ResidualWrapperBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'residual_fn' in config:
            config = config.copy()
            residual_function = _parse_config_to_function(config, custom_objects, 'residual_fn', 'residual_fn_type', 'residual_fn_module')
            config['residual_fn'] = residual_function
        return super(ResidualWrapperBase, cls).from_config(config, custom_objects=custom_objects)
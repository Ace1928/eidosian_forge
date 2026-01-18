import collections
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.legacy_tf_layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class _RNNCellWrapperV1(RNNCell):
    """Base class for cells wrappers V1 compatibility.

  This class along with `_RNNCellWrapperV2` allows to define cells wrappers that
  are compatible with V1 and V2, and defines helper methods for this purpose.
  """

    def __init__(self, cell, *args, **kwargs):
        super(_RNNCellWrapperV1, self).__init__(*args, **kwargs)
        assert_like_rnncell('cell', cell)
        self.cell = cell
        if isinstance(cell, trackable.Trackable):
            self._track_trackable(self.cell, name='cell')

    def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
        """Calls the wrapped cell and performs the wrapping logic.

    This method is called from the wrapper's `call` or `__call__` methods.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments.

    Returns:
      A pair containing:
      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
        raise NotImplementedError

    def __call__(self, inputs, state, scope=None):
        """Runs the RNN cell step computation.

    We assume that the wrapped RNNCell is being built within its `__call__`
    method. We directly use the wrapped cell's `__call__` in the overridden
    wrapper `__call__` method.

    This allows to use the wrapped cell and the non-wrapped cell equivalently
    when using `__call__`.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      scope: VariableScope for the subgraph created in the wrapped cells'
        `__call__`.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
        return self._call_wrapped_cell(inputs, state, cell_call_fn=self.cell.__call__, scope=scope)

    def get_config(self):
        config = {'cell': {'class_name': self.cell.__class__.__name__, 'config': self.cell.get_config()}}
        base_config = super(_RNNCellWrapperV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        cell = config.pop('cell')
        try:
            assert_like_rnncell('cell', cell)
            return cls(cell, **config)
        except TypeError:
            raise ValueError('RNNCellWrapper cannot reconstruct the wrapped cell. Please overwrite the cell in the config with a RNNCell instance.')
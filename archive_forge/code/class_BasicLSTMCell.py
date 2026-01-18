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
@tf_export(v1=['nn.rnn_cell.BasicLSTMCell'])
class BasicLSTMCell(LayerRNNCell):
    """DEPRECATED: Please use `tf.compat.v1.nn.rnn_cell.LSTMCell` instead.

  Basic LSTM recurrent network cell.

  The implementation is based on

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full `tf.compat.v1.nn.rnn_cell.LSTMCell`
  that follows.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above). Must set
        to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases.
      dtype: Default dtype of the layer (default of `None` means use the type of
        the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
        When restoring from CudnnLSTM-trained checkpoints, must use
        `CudnnCompatibleLSTMCell` instead.
    """
        warnings.warn('`tf.nn.rnn_cell.BasicLSTMCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.LSTMCell`, and will be replaced by that in Tensorflow 2.0.')
        super(BasicLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if not state_is_tuple:
            logging.warning('%s: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.', self)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnLSTM for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[4 * self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped `[batch_size,
        num_units]`, if `state_is_tuple` has been set to `True`.  Otherwise, a
        `Tensor` shaped `[batch_size, 2 * num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
        _check_rnn_cell_input_dtypes([inputs, state])
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return (new_h, new_state)

    def get_config(self):
        config = {'num_units': self._num_units, 'forget_bias': self._forget_bias, 'state_is_tuple': self._state_is_tuple, 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(BasicLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
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
class SimpleRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for SimpleRNN.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.SimpleRNN` processes the whole sequence.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(4))

  output = rnn(inputs)  # The output has shape `[32, 4]`.

  rnn = tf.keras.layers.RNN(
      tf.keras.layers.SimpleRNNCell(4),
      return_sequences=True,
      return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = rnn(inputs)
  ```
  """

    def __init__(self, units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for units, expected a positive integer, got {units}.')
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(SimpleRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), name='kernel', initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='recurrent_kernel', initializer=self.recurrent_initializer, regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint, caching_device=default_caching_device)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), name='bias', initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_nested(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)
        if dp_mask is not None:
            h = backend.dot(inputs * dp_mask, self.kernel)
        else:
            h = backend.dot(inputs, self.kernel)
        if self.bias is not None:
            h = backend.bias_add(h, self.bias)
        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)
        new_state = [output] if nest.is_nested(states) else output
        return (output, new_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {'units': self.units, 'activation': activations.serialize(self.activation), 'use_bias': self.use_bias, 'kernel_initializer': initializers.serialize(self.kernel_initializer), 'recurrent_initializer': initializers.serialize(self.recurrent_initializer), 'bias_initializer': initializers.serialize(self.bias_initializer), 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer), 'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer), 'bias_regularizer': regularizers.serialize(self.bias_regularizer), 'kernel_constraint': constraints.serialize(self.kernel_constraint), 'recurrent_constraint': constraints.serialize(self.recurrent_constraint), 'bias_constraint': constraints.serialize(self.bias_constraint), 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout}
        config.update(_config_for_enable_caching_device(self))
        base_config = super(SimpleRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
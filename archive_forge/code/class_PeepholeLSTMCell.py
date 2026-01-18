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
class PeepholeLSTMCell(LSTMCell):
    """Equivalent to LSTMCell class but adds peephole connections.

  Peephole connections allow the gates to utilize the previous internal state as
  well as the previous hidden state (which is what LSTMCell is limited to).
  This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

  From [Gers et al., 2002](
    http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

  "We find that LSTM augmented by 'peephole connections' from its internal
  cells to its multiplicative gates can learn the fine distinction between
  sequences of spikes spaced either 50 or 49 time steps apart without the help
  of any short training exemplars."

  The peephole implementation is based on:

  [Sak et al., 2014](https://research.google.com/pubs/archive/43905.pdf)

  Example:

  ```python
  # Create 2 PeepholeLSTMCells
  peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
  # Create a layer composed sequentially of the peephole LSTM cells.
  layer = RNN(peephole_lstm_cells)
  input = keras.Input((timesteps, input_dim))
  output = layer(input)
  ```
  """

    def __init__(self, units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, **kwargs):
        warnings.warn('`tf.keras.experimental.PeepholeLSTMCell` is deprecated and will be removed in a future version. Please use tensorflow_addons.rnn.PeepholeLSTMCell instead.')
        super(PeepholeLSTMCell, self).__init__(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=kwargs.pop('implementation', 1), **kwargs)

    def build(self, input_shape):
        super(PeepholeLSTMCell, self).build(input_shape)
        self.input_gate_peephole_weights = self.add_weight(shape=(self.units,), name='input_gate_peephole_weights', initializer=self.kernel_initializer)
        self.forget_gate_peephole_weights = self.add_weight(shape=(self.units,), name='forget_gate_peephole_weights', initializer=self.kernel_initializer)
        self.output_gate_peephole_weights = self.add_weight(shape=(self.units,), name='output_gate_peephole_weights', initializer=self.kernel_initializer)

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) + self.input_gate_peephole_weights * c_tm1)
        f = self.recurrent_activation(x_f + backend.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + self.forget_gate_peephole_weights * c_tm1)
        c = f * c_tm1 + i * self.activation(x_c + backend.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + self.output_gate_peephole_weights * c)
        return (c, o)

    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0 + self.input_gate_peephole_weights * c_tm1)
        f = self.recurrent_activation(z1 + self.forget_gate_peephole_weights * c_tm1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
        return (c, o)
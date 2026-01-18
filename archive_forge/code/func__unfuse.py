import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
def _unfuse(self):
    """Unfuses the fused RNN in to a stack of rnn cells."""
    assert not self._projection_size, '_unfuse does not support projection layer yet!'
    assert not self._lstm_state_clip_min and (not self._lstm_state_clip_max), '_unfuse does not support state clipping yet!'
    get_cell = {'rnn_relu': lambda **kwargs: rnn_cell.RNNCell(self._hidden_size, activation='relu', **kwargs), 'rnn_tanh': lambda **kwargs: rnn_cell.RNNCell(self._hidden_size, activation='tanh', **kwargs), 'lstm': lambda **kwargs: rnn_cell.LSTMCell(self._hidden_size, **kwargs), 'gru': lambda **kwargs: rnn_cell.GRUCell(self._hidden_size, **kwargs)}[self._mode]
    stack = rnn_cell.HybridSequentialRNNCell(prefix=self.prefix, params=self.params)
    with stack.name_scope():
        ni = self._input_size
        for i in range(self._num_layers):
            kwargs = {'input_size': ni, 'i2h_weight_initializer': self._i2h_weight_initializer, 'h2h_weight_initializer': self._h2h_weight_initializer, 'i2h_bias_initializer': self._i2h_bias_initializer, 'h2h_bias_initializer': self._h2h_bias_initializer}
            if self._dir == 2:
                stack.add(rnn_cell.BidirectionalCell(get_cell(prefix='l%d_' % i, **kwargs), get_cell(prefix='r%d_' % i, **kwargs)))
            else:
                stack.add(get_cell(prefix='l%d_' % i, **kwargs))
            if self._dropout > 0 and i != self._num_layers - 1:
                stack.add(rnn_cell.DropoutCell(self._dropout))
            ni = self._hidden_size * self._dir
    return stack
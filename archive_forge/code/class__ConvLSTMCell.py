from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class _ConvLSTMCell(_BaseConvRNNCell):

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, dims, conv_layout, activation, prefix, params):
        super(_ConvLSTMCell, self).__init__(input_shape=input_shape, hidden_channels=hidden_channels, i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, dims=dims, conv_layout=conv_layout, activation=activation, prefix=prefix, params=params)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,) + self._state_shape, '__layout__': self._conv_layout}, {'shape': (batch_size,) + self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_lstm'

    @property
    def _gate_names(self):
        return ['_i', '_f', '_c', '_o']

    def hybrid_forward(self, F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        prefix = 't%d_' % self._counter
        i2h, h2h = self._conv_forward(F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, prefix)
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix + 'slice', axis=self._channel_axis)
        in_gate = F.Activation(slice_gates[0], act_type='sigmoid', name=prefix + 'i')
        forget_gate = F.Activation(slice_gates[1], act_type='sigmoid', name=prefix + 'f')
        in_transform = self._get_activation(F, slice_gates[2], self._activation, name=prefix + 'c')
        out_gate = F.Activation(slice_gates[3], act_type='sigmoid', name=prefix + 'o')
        next_c = F.elemwise_add(forget_gate * states[1], in_gate * in_transform, name=prefix + 'state')
        next_h = F.elemwise_mul(out_gate, self._get_activation(F, next_c, self._activation), name=prefix + 'out')
        return (next_h, [next_h, next_c])
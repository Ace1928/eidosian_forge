from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class _ConvGRUCell(_BaseConvRNNCell):

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, dims, conv_layout, activation, prefix, params):
        super(_ConvGRUCell, self).__init__(input_shape=input_shape, hidden_channels=hidden_channels, i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, dims=dims, conv_layout=conv_layout, activation=activation, prefix=prefix, params=params)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,) + self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_gru'

    @property
    def _gate_names(self):
        return ['_r', '_z', '_o']

    def hybrid_forward(self, F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        prefix = 't%d_' % self._counter
        i2h, h2h = self._conv_forward(F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, prefix)
        i2h_r, i2h_z, i2h = F.SliceChannel(i2h, num_outputs=3, name=prefix + 'i2h_slice', axis=self._channel_axis)
        h2h_r, h2h_z, h2h = F.SliceChannel(h2h, num_outputs=3, name=prefix + 'h2h_slice', axis=self._channel_axis)
        reset_gate = F.Activation(i2h_r + h2h_r, act_type='sigmoid', name=prefix + 'r_act')
        update_gate = F.Activation(i2h_z + h2h_z, act_type='sigmoid', name=prefix + 'z_act')
        next_h_tmp = self._get_activation(F, i2h + reset_gate * h2h, self._activation, name=prefix + 'h_act')
        next_h = F.elemwise_add((1.0 - update_gate) * next_h_tmp, update_gate * states[0], name=prefix + 'out')
        return (next_h, [next_h])
from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
def _decide_shapes(self):
    channel_axis = self._conv_layout.find('C')
    input_shape = self._input_shape
    in_channels = input_shape[channel_axis - 1]
    hidden_channels = self._hidden_channels
    if channel_axis == 1:
        dimensions = input_shape[1:]
    else:
        dimensions = input_shape[:-1]
    total_out = hidden_channels * self._num_gates
    i2h_param_shape = (total_out,)
    h2h_param_shape = (total_out,)
    state_shape = (hidden_channels,)
    conv_out_size = _get_conv_out_size(dimensions, self._i2h_kernel, self._i2h_pad, self._i2h_dilate)
    h2h_pad = tuple((d * (k - 1) // 2 for d, k in zip(self._h2h_dilate, self._h2h_kernel)))
    if channel_axis == 1:
        i2h_param_shape += (in_channels,) + self._i2h_kernel
        h2h_param_shape += (hidden_channels,) + self._h2h_kernel
        state_shape += conv_out_size
    else:
        i2h_param_shape += self._i2h_kernel + (in_channels,)
        h2h_param_shape += self._h2h_kernel + (hidden_channels,)
        state_shape = conv_out_size + state_shape
    return (channel_axis, in_channels, i2h_param_shape, h2h_param_shape, h2h_pad, state_shape)
from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class _BaseConvRNNCell(HybridRecurrentCell):
    """Abstract base class for convolutional RNNs"""

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, dims, conv_layout, activation, prefix=None, params=None):
        super(_BaseConvRNNCell, self).__init__(prefix=prefix, params=params)
        self._hidden_channels = hidden_channels
        self._input_shape = input_shape
        self._conv_layout = conv_layout
        self._activation = activation
        assert all((isinstance(spec, int) or len(spec) == dims for spec in [i2h_kernel, i2h_pad, i2h_dilate, h2h_kernel, h2h_dilate])), 'For {dims}D convolution, the convolution settings can only be either int or list/tuple of length {dims}'.format(dims=dims)
        self._i2h_kernel = (i2h_kernel,) * dims if isinstance(i2h_kernel, numeric_types) else i2h_kernel
        self._stride = (1,) * dims
        self._i2h_pad = (i2h_pad,) * dims if isinstance(i2h_pad, numeric_types) else i2h_pad
        self._i2h_dilate = (i2h_dilate,) * dims if isinstance(i2h_dilate, numeric_types) else i2h_dilate
        self._h2h_kernel = (h2h_kernel,) * dims if isinstance(h2h_kernel, numeric_types) else h2h_kernel
        assert all((k % 2 == 1 for k in self._h2h_kernel)), 'Only support odd number, get h2h_kernel= %s' % str(h2h_kernel)
        self._h2h_dilate = (h2h_dilate,) * dims if isinstance(h2h_dilate, numeric_types) else h2h_dilate
        self._channel_axis, self._in_channels, i2h_param_shape, h2h_param_shape, self._h2h_pad, self._state_shape = self._decide_shapes()
        self.i2h_weight = self.params.get('i2h_weight', shape=i2h_param_shape, init=i2h_weight_initializer, allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=h2h_param_shape, init=h2h_weight_initializer, allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(hidden_channels * self._num_gates,), init=i2h_bias_initializer, allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(hidden_channels * self._num_gates,), init=h2h_bias_initializer, allow_deferred_init=True)

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

    def __repr__(self):
        s = '{name}({mapping}'
        if hasattr(self, '_activation'):
            s += ', {_activation}'
        s += ', {_conv_layout}'
        s += ')'
        attrs = self.__dict__
        shape = self.i2h_weight.shape
        in_channels = shape[1 if self._channel_axis == 1 else -1]
        mapping = '{0} -> {1}'.format(in_channels if in_channels else None, shape[0])
        return s.format(name=self.__class__.__name__, mapping=mapping, **attrs)

    @property
    def _num_gates(self):
        return len(self._gate_names)

    def _conv_forward(self, F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, prefix):
        i2h = F.Convolution(data=inputs, num_filter=self._hidden_channels * self._num_gates, kernel=self._i2h_kernel, stride=self._stride, pad=self._i2h_pad, dilate=self._i2h_dilate, weight=i2h_weight, bias=i2h_bias, layout=self._conv_layout, name=prefix + 'i2h')
        h2h = F.Convolution(data=states[0], num_filter=self._hidden_channels * self._num_gates, kernel=self._h2h_kernel, dilate=self._h2h_dilate, pad=self._h2h_pad, stride=self._stride, weight=h2h_weight, bias=h2h_bias, layout=self._conv_layout, name=prefix + 'h2h')
        return (i2h, h2h)

    def state_info(self, batch_size=0):
        raise NotImplementedError('_BaseConvRNNCell is abstract class for convolutional RNN')

    def hybrid_forward(self, F, inputs, states):
        raise NotImplementedError('_BaseConvRNNCell is abstract class for convolutional RNN')
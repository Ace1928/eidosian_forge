import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
class _RNNLayer(HybridBlock):
    """Implementation of recurrent layers."""

    def __init__(self, hidden_size, num_layers, layout, dropout, bidirectional, input_size, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, mode, projection_size, h2r_weight_initializer, lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan, dtype, use_sequence_length=False, **kwargs):
        super(_RNNLayer, self).__init__(**kwargs)
        assert layout in ('TNC', 'NTC'), "Invalid layout %s; must be one of ['TNC' or 'NTC']" % layout
        self._hidden_size = hidden_size
        self._projection_size = projection_size if projection_size else None
        self._num_layers = num_layers
        self._mode = mode
        self._layout = layout
        self._dropout = dropout
        self._dir = 2 if bidirectional else 1
        self._input_size = input_size
        self._i2h_weight_initializer = i2h_weight_initializer
        self._h2h_weight_initializer = h2h_weight_initializer
        self._i2h_bias_initializer = i2h_bias_initializer
        self._h2h_bias_initializer = h2h_bias_initializer
        self._h2r_weight_initializer = h2r_weight_initializer
        self._lstm_state_clip_min = lstm_state_clip_min
        self._lstm_state_clip_max = lstm_state_clip_max
        self._lstm_state_clip_nan = lstm_state_clip_nan
        self._dtype = dtype
        self._use_sequence_length = use_sequence_length
        self.skip_states = None
        self._gates = {'rnn_relu': 1, 'rnn_tanh': 1, 'lstm': 4, 'gru': 3}[mode]
        ng, ni, nh = (self._gates, input_size, hidden_size)
        if not projection_size:
            for i in range(num_layers):
                for j in ['l', 'r'][:self._dir]:
                    self._register_param('{}{}_i2h_weight'.format(j, i), shape=(ng * nh, ni), init=i2h_weight_initializer, dtype=dtype)
                    self._register_param('{}{}_h2h_weight'.format(j, i), shape=(ng * nh, nh), init=h2h_weight_initializer, dtype=dtype)
                    self._register_param('{}{}_i2h_bias'.format(j, i), shape=(ng * nh,), init=i2h_bias_initializer, dtype=dtype)
                    self._register_param('{}{}_h2h_bias'.format(j, i), shape=(ng * nh,), init=h2h_bias_initializer, dtype=dtype)
                ni = nh * self._dir
        else:
            np = self._projection_size
            for i in range(num_layers):
                for j in ['l', 'r'][:self._dir]:
                    self._register_param('{}{}_i2h_weight'.format(j, i), shape=(ng * nh, ni), init=i2h_weight_initializer, dtype=dtype)
                    self._register_param('{}{}_h2h_weight'.format(j, i), shape=(ng * nh, np), init=h2h_weight_initializer, dtype=dtype)
                    self._register_param('{}{}_i2h_bias'.format(j, i), shape=(ng * nh,), init=i2h_bias_initializer, dtype=dtype)
                    self._register_param('{}{}_h2h_bias'.format(j, i), shape=(ng * nh,), init=h2h_bias_initializer, dtype=dtype)
                    self._register_param('{}{}_h2r_weight'.format(j, i), shape=(np, nh), init=h2r_weight_initializer, dtype=dtype)
                ni = np * self._dir

    def _register_param(self, name, shape, init, dtype):
        p = self.params.get(name, shape=shape, init=init, allow_deferred_init=True, dtype=dtype)
        setattr(self, name, p)
        return p

    def __repr__(self):
        s = '{name}({mapping}, {_layout}'
        if self._num_layers != 1:
            s += ', num_layers={_num_layers}'
        if self._dropout != 0:
            s += ', dropout={_dropout}'
        if self._dir == 2:
            s += ', bidirectional'
        s += ')'
        shape = self.l0_i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0] // self._gates)
        return s.format(name=self.__class__.__name__, mapping=mapping, **self.__dict__)

    def _collect_params_with_prefix(self, prefix=''):
        if prefix:
            prefix += '.'
        pattern = re.compile('(l|r)(\\d)_(i2h|h2h|h2r)_(weight|bias)\\Z')

        def convert_key(m, bidirectional):
            d, l, g, t = [m.group(i) for i in range(1, 5)]
            if bidirectional:
                return '_unfused.{}.{}_cell.{}_{}'.format(l, d, g, t)
            else:
                return '_unfused.{}.{}_{}'.format(l, g, t)
        bidirectional = any((pattern.match(k).group(1) == 'r' for k in self._reg_params))
        ret = {prefix + convert_key(pattern.match(key), bidirectional): val for key, val in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    def state_info(self, batch_size=0):
        raise NotImplementedError

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

    def cast(self, dtype):
        super(_RNNLayer, self).cast(dtype)
        self._dtype = dtype

    def begin_state(self, batch_size=0, func=ndarray.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        batch_size: int
            Only required for `NDArray` API. Size of the batch ('N' in layout).
            Dimension of the input.
        func : callable, default `ndarray.zeros`
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var` etc. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        states = []
        for i, info in enumerate(self.state_info(batch_size)):
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(name='%sh0_%d' % (self.prefix, i), **info)
            if is_np_array():
                state = state.as_np_ndarray()
            states.append(state)
        return states

    def __call__(self, inputs, states=None, sequence_length=None, **kwargs):
        self.skip_states = states is None
        if states is None:
            if isinstance(inputs, ndarray.NDArray):
                batch_size = inputs.shape[self._layout.find('N')]
                states = self.begin_state(batch_size, ctx=inputs.context, dtype=inputs.dtype)
            else:
                states = self.begin_state(0, func=symbol.zeros)
        if isinstance(states, tensor_types):
            states = [states]
        if self._use_sequence_length:
            return super(_RNNLayer, self).__call__(inputs, states, sequence_length, **kwargs)
        else:
            return super(_RNNLayer, self).__call__(inputs, states, **kwargs)

    def hybrid_forward(self, F, inputs, states, sequence_length=None, **kwargs):
        if F is ndarray:
            batch_size = inputs.shape[self._layout.find('N')]
        if F is ndarray:
            for state, info in zip(states, self.state_info(batch_size)):
                if state.shape != info['shape']:
                    raise ValueError('Invalid recurrent state shape. Expecting %s, got %s.' % (str(info['shape']), str(state.shape)))
        out = self._forward_kernel(F, inputs, states, sequence_length, **kwargs)
        return out[0] if self.skip_states else out

    def _forward_kernel(self, F, inputs, states, sequence_length, **kwargs):
        """ forward using CUDNN or CPU kenrel"""
        swapaxes = F.np.swapaxes if is_np_array() else F.swapaxes
        if self._layout == 'NTC':
            inputs = swapaxes(inputs, 0, 1)
        if self._projection_size is None:
            params = (kwargs['{}{}_{}_{}'.format(d, l, g, t)].reshape(-1) for t in ['weight', 'bias'] for l in range(self._num_layers) for d in ['l', 'r'][:self._dir] for g in ['i2h', 'h2h'])
        else:
            params = (kwargs['{}{}_{}_{}'.format(d, l, g, t)].reshape(-1) for t in ['weight', 'bias'] for l in range(self._num_layers) for d in ['l', 'r'][:self._dir] for g in ['i2h', 'h2h', 'h2r'] if g != 'h2r' or t != 'bias')
        rnn_param_concat = F.np._internal.rnn_param_concat if is_np_array() else F._internal._rnn_param_concat
        params = rnn_param_concat(*params, dim=0)
        if self._use_sequence_length:
            rnn_args = states + [sequence_length]
        else:
            rnn_args = states
        rnn_fn = F.npx.rnn if is_np_array() else F.RNN
        rnn = rnn_fn(inputs, params, *rnn_args, use_sequence_length=self._use_sequence_length, state_size=self._hidden_size, projection_size=self._projection_size, num_layers=self._num_layers, bidirectional=self._dir == 2, p=self._dropout, state_outputs=True, mode=self._mode, lstm_state_clip_min=self._lstm_state_clip_min, lstm_state_clip_max=self._lstm_state_clip_max, lstm_state_clip_nan=self._lstm_state_clip_nan)
        if self._mode == 'lstm':
            outputs, states = (rnn[0], [rnn[1], rnn[2]])
        else:
            outputs, states = (rnn[0], [rnn[1]])
        if self._layout == 'NTC':
            outputs = swapaxes(outputs, 0, 1)
        return (outputs, states)
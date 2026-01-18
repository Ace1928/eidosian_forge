import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
class FusedRNNCell(BaseRNNCell):
    """Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.

    Parameters
    ----------
    num_hidden : int
        Number of units in output symbol.
    num_layers : int, default 1
        Number of layers in the cell.
    mode : str, default 'lstm'
        Type of RNN. options are 'rnn_relu', 'rnn_tanh', 'lstm', 'gru'.
    bidirectional : bool, default False
        Whether to use bidirectional unroll. The output dimension size is doubled if bidrectional.
    dropout : float, default 0.
        Fraction of the input that gets dropped out during training time.
    get_next_state : bool, default False
        Whether to return the states that can be used as starting states next time.
    forget_bias : bias added to forget gate, default 1.0.
        Jozefowicz et al. 2015 recommends setting this to 1.0
    prefix : str, default ``'$mode_'`` such as ``'lstm_'``
        Prefix for names of layers
        (this prefix is also used for names of weights if `params` is None
        i.e. if `params` are being created and not reused)
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """

    def __init__(self, num_hidden, num_layers=1, mode='lstm', bidirectional=False, dropout=0.0, get_next_state=False, forget_bias=1.0, prefix=None, params=None):
        if prefix is None:
            prefix = '%s_' % mode
        super(FusedRNNCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._mode = mode
        self._bidirectional = bidirectional
        self._dropout = dropout
        self._get_next_state = get_next_state
        self._directions = ['l', 'r'] if bidirectional else ['l']
        initializer = init.FusedRNN(None, num_hidden, num_layers, mode, bidirectional, forget_bias)
        self._parameter = self.params.get('parameters', init=initializer)

    @property
    def state_info(self):
        b = self._bidirectional + 1
        n = (self._mode == 'lstm') + 1
        return [{'shape': (b * self._num_layers, 0, self._num_hidden), '__layout__': 'LNC'} for _ in range(n)]

    @property
    def _gate_names(self):
        return {'rnn_relu': [''], 'rnn_tanh': [''], 'lstm': ['_i', '_f', '_c', '_o'], 'gru': ['_r', '_z', '_o']}[self._mode]

    @property
    def _num_gates(self):
        return len(self._gate_names)

    def _slice_weights(self, arr, li, lh):
        """slice fused rnn weights"""
        args = {}
        gate_names = self._gate_names
        directions = self._directions
        b = len(directions)
        p = 0
        for layer in range(self._num_layers):
            for direction in directions:
                for gate in gate_names:
                    name = '%s%s%d_i2h%s_weight' % (self._prefix, direction, layer, gate)
                    if layer > 0:
                        size = b * lh * lh
                        args[name] = arr[p:p + size].reshape((lh, b * lh))
                    else:
                        size = li * lh
                        args[name] = arr[p:p + size].reshape((lh, li))
                    p += size
                for gate in gate_names:
                    name = '%s%s%d_h2h%s_weight' % (self._prefix, direction, layer, gate)
                    size = lh ** 2
                    args[name] = arr[p:p + size].reshape((lh, lh))
                    p += size
        for layer in range(self._num_layers):
            for direction in directions:
                for gate in gate_names:
                    name = '%s%s%d_i2h%s_bias' % (self._prefix, direction, layer, gate)
                    args[name] = arr[p:p + lh]
                    p += lh
                for gate in gate_names:
                    name = '%s%s%d_h2h%s_bias' % (self._prefix, direction, layer, gate)
                    args[name] = arr[p:p + lh]
                    p += lh
        assert p == arr.size, 'Invalid parameters size for FusedRNNCell'
        return args

    def unpack_weights(self, args):
        args = args.copy()
        arr = args.pop(self._parameter.name)
        b = len(self._directions)
        m = self._num_gates
        h = self._num_hidden
        num_input = arr.size // b // h // m - (self._num_layers - 1) * (h + b * h + 2) - h - 2
        nargs = self._slice_weights(arr, num_input, self._num_hidden)
        args.update({name: nd.copy() for name, nd in nargs.items()})
        return args

    def pack_weights(self, args):
        args = args.copy()
        b = self._bidirectional + 1
        m = self._num_gates
        c = self._gate_names
        h = self._num_hidden
        w0 = args['%sl0_i2h%s_weight' % (self._prefix, c[0])]
        num_input = w0.shape[1]
        total = (num_input + h + 2) * h * m * b + (self._num_layers - 1) * m * h * (h + b * h + 2) * b
        arr = ndarray.zeros((total,), ctx=w0.context, dtype=w0.dtype)
        for name, nd in self._slice_weights(arr, num_input, h).items():
            nd[:] = args.pop(name)
        args[self._parameter.name] = arr
        return args

    def __call__(self, inputs, states):
        raise NotImplementedError('FusedRNNCell cannot be stepped. Please use unroll')

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()
        inputs, axis = _normalize_sequence(length, inputs, layout, True)
        if axis == 1:
            warnings.warn('NTC layout detected. Consider using TNC for FusedRNNCell for faster speed')
            inputs = symbol.swapaxes(inputs, dim1=0, dim2=1)
        else:
            assert axis == 0, 'Unsupported layout %s' % layout
        if begin_state is None:
            begin_state = self.begin_state()
        states = begin_state
        if self._mode == 'lstm':
            states = {'state': states[0], 'state_cell': states[1]}
        else:
            states = {'state': states[0]}
        rnn = symbol.RNN(data=inputs, parameters=self._parameter, state_size=self._num_hidden, num_layers=self._num_layers, bidirectional=self._bidirectional, p=self._dropout, state_outputs=self._get_next_state, mode=self._mode, name=self._prefix + 'rnn', **states)
        attr = {'__layout__': 'LNC'}
        if not self._get_next_state:
            outputs, states = (rnn, [])
        elif self._mode == 'lstm':
            rnn[1]._set_attr(**attr)
            rnn[2]._set_attr(**attr)
            outputs, states = (rnn[0], [rnn[1], rnn[2]])
        else:
            rnn[1]._set_attr(**attr)
            outputs, states = (rnn[0], [rnn[1]])
        if axis == 1:
            outputs = symbol.swapaxes(outputs, dim1=0, dim2=1)
        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)
        return (outputs, states)

    def unfuse(self):
        """Unfuse the fused RNN in to a stack of rnn cells.

        Returns
        -------
        cell : mxnet.rnn.SequentialRNNCell
            unfused cell that can be used for stepping, and can run on CPU.
        """
        stack = SequentialRNNCell()
        get_cell = {'rnn_relu': lambda cell_prefix: RNNCell(self._num_hidden, activation='relu', prefix=cell_prefix), 'rnn_tanh': lambda cell_prefix: RNNCell(self._num_hidden, activation='tanh', prefix=cell_prefix), 'lstm': lambda cell_prefix: LSTMCell(self._num_hidden, prefix=cell_prefix), 'gru': lambda cell_prefix: GRUCell(self._num_hidden, prefix=cell_prefix)}[self._mode]
        for i in range(self._num_layers):
            if self._bidirectional:
                stack.add(BidirectionalCell(get_cell('%sl%d_' % (self._prefix, i)), get_cell('%sr%d_' % (self._prefix, i)), output_prefix='%sbi_l%d_' % (self._prefix, i)))
            else:
                stack.add(get_cell('%sl%d_' % (self._prefix, i)))
            if self._dropout > 0 and i != self._num_layers - 1:
                stack.add(DropoutCell(self._dropout, prefix='%s_dropout%d_' % (self._prefix, i)))
        return stack
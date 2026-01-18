import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
class BaseRNNCell(object):
    """Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str, optional
        Prefix for names of layers
        (this prefix is also used for names of weights if `params` is None
        i.e. if `params` are being created and not reused)
    params : RNNParams, default None.
        Container for weight sharing between cells.
        A new RNNParams container is created if `params` is None.
    """

    def __init__(self, prefix='', params=None):
        if params is None:
            params = RNNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._prefix = prefix
        self._params = params
        self._modified = False
        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1
        if hasattr(self, '_cells'):
            for cell in self._cells:
                cell.reset()

    def __call__(self, inputs, states):
        """Unroll the RNN for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
            This can be used as input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Parameters of this cell"""
        self._own_params = False
        return self._params

    @property
    def state_info(self):
        """shape and layout information of states"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [ele['shape'] for ele in self.state_info]

    @property
    def _gate_names(self):
        """name(s) of gates"""
        return ()

    def begin_state(self, func=symbol.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state. Can be symbol.zeros,
            symbol.uniform, symbol.Variable etc.
            Use symbol.Variable if you want to directly
            feed input as states.
        **kwargs :
            more keyword arguments passed to func. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        assert not self._modified, 'After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
        states = []
        for info in self.state_info:
            self._init_counter += 1
            if info is None:
                state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter), **kwargs)
            else:
                kwargs.update(info)
                state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter), **kwargs)
            states.append(state)
        return states

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices.

        For example, say you use a module object `mod` to run a network that has an lstm cell.
        In `mod.get_params()[0]`, the lstm parameters are all represented as a single big vector.
        `cell.unpack_weights(mod.get_params()[0])` will unpack this vector into a dictionary of
        more readable lstm parameters - c, f, i, o gates for i2h (input to hidden) and
        h2h (hidden to hidden) weights.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing packed weights.
            usually from `Module.get_params()[0]`.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with unpacked weights associated with
            this cell.

        See Also
        --------
        pack_weights: Performs the reverse operation of this function.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        h = self._num_hidden
        for group_name in ['i2h', 'h2h']:
            weight = args.pop('%s%s_weight' % (self._prefix, group_name))
            bias = args.pop('%s%s_bias' % (self._prefix, group_name))
            for j, gate in enumerate(self._gate_names):
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                args[wname] = weight[j * h:(j + 1) * h].copy()
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                args[bname] = bias[j * h:(j + 1) * h].copy()
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into a single packed
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with packed weights associated with
            this cell.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        for group_name in ['i2h', 'h2h']:
            weight = []
            bias = []
            for gate in self._gate_names:
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                weight.append(args.pop(wname))
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                bias.append(args.pop(bname))
            args['%s%s_weight' % (self._prefix, group_name)] = ndarray.concatenate(weight)
            args['%s%s_bias' % (self._prefix, group_name)] = ndarray.concatenate(bias)
        return args

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, default None
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if None.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
        """
        self.reset()
        inputs, _ = _normalize_sequence(length, inputs, layout, False)
        if begin_state is None:
            begin_state = self.begin_state()
        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)
        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)
        return (outputs, states)

    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return symbol.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)
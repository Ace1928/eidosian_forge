from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class RecurrentCell(Block):
    """Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str, optional
        Prefix for names of `Block`s
        (this prefix is also used for names of weights if `params` is `None`
        i.e. if `params` are being created and not reused)
    params : Parameter or None, default None
        Container for weight sharing between cells.
        A new Parameter container is created if `params` is `None`.
    """

    def __init__(self, prefix=None, params=None):
        super(RecurrentCell, self).__init__(prefix=prefix, params=params)
        self._modified = False
        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1
        for cell in self._children.values():
            cell.reset()

    def state_info(self, batch_size=0):
        """shape and layout information of states"""
        raise NotImplementedError()

    def begin_state(self, batch_size=0, func=ndarray.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var etc`. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.
        batch_size: int, default 0
            Only required for NDArray API. Size of the batch ('N' in layout)
            dimension of input.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        assert not self._modified, 'After applying modifier cells (e.g. ZoneoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
        states = []
        for info in self.state_info(batch_size):
            self._init_counter += 1
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter), **info)
            states.append(state)
        return states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None, valid_length=None):
        """Unrolls an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if `layout` is 'NTC',
            or (length, batch_size, ...) if `layout` is 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if `None`.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If `False`, returns outputs as a list of Symbols.
            If `True`, concatenates output across time steps
            and returns a single symbol with shape
            (batch_size, length, ...) if layout is 'NTC',
            or (length, batch_size, ...) if layout is 'TNC'.
            If `None`, output whatever is faster.
        valid_length : Symbol, NDArray or None
            `valid_length` specifies the length of the sequences in the batch without padding.
            This option is especially useful for building sequence-to-sequence models where
            the input and output sequences would potentially be padded.
            If `valid_length` is None, all sequences are assumed to have the same length.
            If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
            The ith element will be the length of the ith sequence in the batch.
            The last valid state will be return and the padded outputs will be masked with 0.
            Note that `valid_length` must be smaller or equal to `length`.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
        """
        self.reset()
        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, False)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)
        states = begin_state
        outputs = []
        all_states = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)
            if valid_length is not None:
                all_states.append(states)
        if valid_length is not None:
            states = [F.SequenceLast(F.stack(*ele_list, axis=0), sequence_length=valid_length, use_sequence_length=True, axis=0) for ele_list in zip(*all_states)]
            outputs = _mask_sequence_variable_length(F, outputs, length, valid_length, axis, True)
        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)
        return (outputs, states)

    def _get_activation(self, F, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        func = {'tanh': F.tanh, 'relu': F.relu, 'sigmoid': F.sigmoid, 'softsign': F.softsign}.get(activation)
        if func:
            return func(inputs, **kwargs)
        elif isinstance(activation, string_types):
            return F.Activation(inputs, act_type=activation, **kwargs)
        elif isinstance(activation, LeakyReLU):
            return F.LeakyReLU(inputs, act_type='leaky', slope=activation._alpha, **kwargs)
        return activation(inputs, **kwargs)

    def forward(self, inputs, states):
        """Unrolls the recurrent cell for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            Input symbol, 2D, of shape (batch_size * num_units).
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
            This can be used as an input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        self._counter += 1
        return super(RecurrentCell, self).forward(inputs, states)
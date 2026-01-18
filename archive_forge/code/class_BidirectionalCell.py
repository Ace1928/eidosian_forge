from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class BidirectionalCell(HybridRecurrentCell):
    """Bidirectional RNN cell.

    Parameters
    ----------
    l_cell : RecurrentCell
        Cell for forward unrolling
    r_cell : RecurrentCell
        Cell for backward unrolling
    """

    def __init__(self, l_cell, r_cell, output_prefix='bi_'):
        super(BidirectionalCell, self).__init__(prefix='', params=None)
        self.register_child(l_cell, 'l_cell')
        self.register_child(r_cell, 'r_cell')
        self._output_prefix = output_prefix

    def __call__(self, inputs, states):
        raise NotImplementedError('Bidirectional cannot be stepped. Please use unroll')

    def __repr__(self):
        s = '{name}(forward={l_cell}, backward={r_cell})'
        return s.format(name=self.__class__.__name__, l_cell=self._children['l_cell'], r_cell=self._children['r_cell'])

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, 'After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
        return _cells_begin_state(self._children.values(), **kwargs)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None, valid_length=None):
        self.reset()
        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, False)
        reversed_inputs = list(_reverse_sequences(inputs, length, valid_length))
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)
        states = begin_state
        l_cell, r_cell = self._children.values()
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs, begin_state=states[:len(l_cell.state_info(batch_size))], layout=layout, merge_outputs=merge_outputs, valid_length=valid_length)
        r_outputs, r_states = r_cell.unroll(length, inputs=reversed_inputs, begin_state=states[len(l_cell.state_info(batch_size)):], layout=layout, merge_outputs=False, valid_length=valid_length)
        reversed_r_outputs = _reverse_sequences(r_outputs, length, valid_length)
        if merge_outputs is None:
            merge_outputs = isinstance(l_outputs, tensor_types)
            l_outputs, _, _, _ = _format_sequence(None, l_outputs, layout, merge_outputs)
            reversed_r_outputs, _, _, _ = _format_sequence(None, reversed_r_outputs, layout, merge_outputs)
        if merge_outputs:
            reversed_r_outputs = F.stack(*reversed_r_outputs, axis=axis)
            outputs = F.concat(l_outputs, reversed_r_outputs, dim=2, name='%sout' % self._output_prefix)
        else:
            outputs = [F.concat(l_o, r_o, dim=1, name='%st%d' % (self._output_prefix, i)) for i, (l_o, r_o) in enumerate(zip(l_outputs, reversed_r_outputs))]
        if valid_length is not None:
            outputs = _mask_sequence_variable_length(F, outputs, length, valid_length, axis, merge_outputs)
        states = l_states + r_states
        return (outputs, states)
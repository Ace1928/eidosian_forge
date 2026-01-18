from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class ResidualCell(ModifierCell):
    """
    Adds residual connection as described in Wu et al, 2016
    (https://arxiv.org/abs/1609.08144).
    Output of the cell is output of the base cell plus input.
    """

    def __init__(self, base_cell):
        super(ResidualCell, self).__init__(base_cell)

    def hybrid_forward(self, F, inputs, states):
        output, states = self.base_cell(inputs, states)
        output = F.elemwise_add(output, inputs, name='t%d_fwd' % self._counter)
        return (output, states)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None, valid_length=None):
        self.reset()
        self.base_cell._modified = False
        outputs, states = self.base_cell.unroll(length, inputs=inputs, begin_state=begin_state, layout=layout, merge_outputs=merge_outputs, valid_length=valid_length)
        self.base_cell._modified = True
        merge_outputs = isinstance(outputs, tensor_types) if merge_outputs is None else merge_outputs
        inputs, axis, F, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if valid_length is not None:
            inputs = _mask_sequence_variable_length(F, inputs, length, valid_length, axis, merge_outputs)
        if merge_outputs:
            outputs = F.elemwise_add(outputs, inputs)
        else:
            outputs = [F.elemwise_add(i, j) for i, j in zip(outputs, inputs)]
        return (outputs, states)
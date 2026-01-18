from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class DropoutCell(HybridRecurrentCell):
    """Applies dropout on input.

    Parameters
    ----------
    rate : float
        Percentage of elements to drop out, which
        is 1 - percentage to retain.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.


    Inputs:
        - **data**: input tensor with shape `(batch_size, size)`.
        - **states**: a list of recurrent state tensors.

    Outputs:
        - **out**: output tensor with shape `(batch_size, size)`.
        - **next_states**: returns input `states` directly.
    """

    def __init__(self, rate, axes=(), prefix=None, params=None):
        super(DropoutCell, self).__init__(prefix, params)
        assert isinstance(rate, numeric_types), 'rate must be a number'
        self._rate = rate
        self._axes = axes

    def __repr__(self):
        s = '{name}(rate={_rate}, axes={_axes})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def state_info(self, batch_size=0):
        return []

    def _alias(self):
        return 'dropout'

    def hybrid_forward(self, F, inputs, states):
        if self._rate > 0:
            inputs = F.Dropout(data=inputs, p=self._rate, axes=self._axes, name='t%d_fwd' % self._counter)
        return (inputs, states)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None, valid_length=None):
        self.reset()
        inputs, _, F, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if isinstance(inputs, tensor_types):
            return self.hybrid_forward(F, inputs, begin_state if begin_state else [])
        return super(DropoutCell, self).unroll(length, inputs, begin_state=begin_state, layout=layout, merge_outputs=merge_outputs, valid_length=None)
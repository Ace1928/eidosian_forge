from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _get_begin_state(cell, F, begin_state, inputs, batch_size):
    if begin_state is None:
        if F is ndarray:
            ctx = inputs.context if isinstance(inputs, tensor_types) else inputs[0].context
            with ctx:
                begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
        else:
            begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
    return begin_state
from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _reverse_sequences(sequences, unroll_step, valid_length=None):
    if isinstance(sequences[0], symbol.Symbol):
        F = symbol
    else:
        F = ndarray
    if valid_length is None:
        reversed_sequences = list(reversed(sequences))
    else:
        reversed_sequences = F.SequenceReverse(F.stack(*sequences, axis=0), sequence_length=valid_length, use_sequence_length=True)
        if unroll_step > 1 or F is symbol:
            reversed_sequences = F.split(reversed_sequences, axis=0, num_outputs=unroll_step, squeeze_axis=True)
        else:
            reversed_sequences = [reversed_sequences[0]]
    return reversed_sequences
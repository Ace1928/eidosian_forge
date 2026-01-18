from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class HybridRecurrentCell(RecurrentCell, HybridBlock):
    """HybridRecurrentCell supports hybridize."""

    def __init__(self, prefix=None, params=None):
        super(HybridRecurrentCell, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError
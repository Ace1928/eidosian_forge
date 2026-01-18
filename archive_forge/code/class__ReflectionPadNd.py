from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F
from torch import Tensor
from ..common_types import _size_2_t, _size_4_t, _size_6_t
from typing import Sequence, Tuple
class _ReflectionPadNd(Module):
    __constants__ = ['padding']
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'reflect')

    def extra_repr(self) -> str:
        return f'{self.padding}'
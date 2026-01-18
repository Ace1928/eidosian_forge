from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class _AdaptiveMaxPoolNd(Module):
    __constants__ = ['output_size', 'return_indices']
    return_indices: bool

    def __init__(self, output_size: _size_any_opt_t, return_indices: bool=False) -> None:
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def extra_repr(self) -> str:
        return f'output_size={self.output_size}'
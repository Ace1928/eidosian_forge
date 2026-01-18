from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t]=None, padding: _size_any_t=0, dilation: _size_any_t=1, return_indices: bool=False, ceil_mode: bool=False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)
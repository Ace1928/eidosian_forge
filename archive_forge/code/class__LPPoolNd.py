from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class _LPPoolNd(Module):
    __constants__ = ['norm_type', 'kernel_size', 'stride', 'ceil_mode']
    norm_type: float
    ceil_mode: bool

    def __init__(self, norm_type: float, kernel_size: _size_any_t, stride: Optional[_size_any_t]=None, ceil_mode: bool=False) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'norm_type={norm_type}, kernel_size={kernel_size}, stride={stride}, ceil_mode={ceil_mode}'.format(**self.__dict__)
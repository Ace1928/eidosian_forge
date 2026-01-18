from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class SharedMem(ArrayBase):

    def __init__(self, child_type: TypeBase, size: Optional[int], alignment: Optional[int]=None) -> None:
        if not (isinstance(size, int) or size is None):
            raise 'size of shared_memory must be integer or `None`'
        if not (isinstance(alignment, int) or alignment is None):
            raise 'alignment must be integer or `None`'
        self._size = size
        self._alignment = alignment
        super().__init__(child_type, 1)

    def declvar(self, x: str, init: Optional['Data']) -> str:
        assert init is None
        if self._alignment is not None:
            code = f'__align__({self._alignment})'
        else:
            code = ''
        if self._size is None:
            code = f'extern {code} __shared__ {self.child_type} {x}[]'
        else:
            code = f'{code} __shared__ {self.child_type} {x}[{self._size}]'
        return code
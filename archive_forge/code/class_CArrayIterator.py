from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class CArrayIterator(PointerBase):

    def __init__(self, carray_type: CArray) -> None:
        self._carray_type = carray_type
        super().__init__(Scalar(carray_type.dtype))

    def __str__(self) -> str:
        return f'{str(self._carray_type)}::iterator'

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TypeBase)
        return isinstance(other, CArrayIterator) and self._carray_type == other._carray_type

    def __hash__(self) -> int:
        return hash((self.dtype, self.ndim, self._c_contiguous, self._index_32_bits))
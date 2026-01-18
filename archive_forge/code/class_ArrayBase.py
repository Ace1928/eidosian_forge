from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class ArrayBase(TypeBase):

    def ndim(self, instance: 'Data'):
        from cupyx.jit import _internal_types
        return _internal_types.Constant(self._ndim)

    def __init__(self, child_type: TypeBase, ndim: int) -> None:
        assert isinstance(child_type, TypeBase)
        self.child_type = child_type
        self._ndim = ndim
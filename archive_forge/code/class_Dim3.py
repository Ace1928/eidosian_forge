from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class Dim3(TypeBase):
    """
    An integer vector type based on uint3 that is used to specify dimensions.

    Attributes:
        x (uint32)
        y (uint32)
        z (uint32)
    """

    def x(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.x', uint32)

    def y(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.y', uint32)

    def z(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.z', uint32)

    def __str__(self) -> str:
        return 'dim3'
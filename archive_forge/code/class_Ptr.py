from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class Ptr(PointerBase):

    def __init__(self, child_type: TypeBase) -> None:
        super().__init__(child_type)

    def __str__(self) -> str:
        return f'{self.child_type}*'
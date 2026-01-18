from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class Unknown(TypeBase):

    def __init__(self, *, label: Optional[str]=None) -> None:
        self.label = label

    def __str__(self) -> str:
        raise TypeError('unknown type can be used only in ary of a function.')
from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class PtrDiff(Scalar):

    def __init__(self) -> None:
        super().__init__('q')

    def __str__(self) -> str:
        return 'ptrdiff_t'
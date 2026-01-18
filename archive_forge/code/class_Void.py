from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class Void(TypeBase):

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return 'void'
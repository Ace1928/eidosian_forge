from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename

    An integer vector type based on uint3 that is used to specify dimensions.

    Attributes:
        x (uint32)
        y (uint32)
        z (uint32)
    
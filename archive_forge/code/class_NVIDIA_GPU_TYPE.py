import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2
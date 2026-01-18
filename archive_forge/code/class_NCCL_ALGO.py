import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1
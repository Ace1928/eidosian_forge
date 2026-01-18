import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
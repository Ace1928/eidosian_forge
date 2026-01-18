import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
class Operand(NamedTuple):
    """Represenation of an NNAPI operand."""
    op_type: int
    shape: Tuple[int, ...]
    dim_order: DimOrder
    scale: float
    zero_point: int

    def use_nchw(self):
        if self.dim_order is DimOrder.PRESUMED_CONTIGUOUS:
            return True
        if self.dim_order is DimOrder.CHANNELS_LAST:
            return False
        raise Exception('Unknown dim order')
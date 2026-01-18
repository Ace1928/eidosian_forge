import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
class ConvPoolArgs2d(NamedTuple):
    """Configuration arguments for a convolution."""
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    dilation_h: int
    dilation_w: int
    group: int
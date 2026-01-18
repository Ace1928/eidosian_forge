import itertools
import numpy
from cupy import _core
from cupy._core import _fusion_interface
from cupy._core import fusion
from cupy._sorting import search
from cupy_backends.cuda.api import runtime
def _can_memcpy(dst, src):
    c_contiguous = dst.flags.c_contiguous and src.flags.c_contiguous
    f_contiguous = dst.flags.f_contiguous and src.flags.f_contiguous
    return (c_contiguous or f_contiguous) and dst.dtype == src.dtype and (dst.size == src.size)
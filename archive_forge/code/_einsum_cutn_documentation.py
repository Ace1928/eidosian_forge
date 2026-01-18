import warnings
import cupy
from cupy._core import _accelerator
Parse & retrieve einsum operands, assuming ``args`` is in either
    "subscript" or "interleaved" format.
    
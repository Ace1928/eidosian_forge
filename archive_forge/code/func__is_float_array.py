import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def _is_float_array(out, *, shape: Optional[Tuple]=None):
    assert out.dtype in ('float32', 'float64'), 'CUDA kernel can only handle float32 and float64'
    if shape is not None and out.shape != shape:
        msg = f'array has incorrect shape, expected: {shape}, was: {out.shape}'
        raise ValueError(msg)
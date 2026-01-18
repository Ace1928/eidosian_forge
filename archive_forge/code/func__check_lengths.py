import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def _check_lengths(lengths, n_elems: int, *, min_length=0):
    assert lengths.dtype == 'int32', 'lengths should be encoded as 32-bit integers'
    if not cupy.all(lengths >= min_length):
        raise ValueError(f'all sequence lengths must be >= {min_length}')
    if cupy.sum(lengths) != n_elems:
        raise IndexError('lengths must sum up to the batch size')
import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def _alloc_like(array, zeros: bool=True):
    if zeros:
        return cupy.zeros_like(array)
    else:
        return cupy.empty_like(array)
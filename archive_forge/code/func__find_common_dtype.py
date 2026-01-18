import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(None)
def _find_common_dtype(array_types, scalar_types):
    import numpy as np
    return np.find_common_type(array_types, scalar_types).name
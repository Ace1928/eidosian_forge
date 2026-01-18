from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@total_ordering
class _RaggedElement:

    @staticmethod
    def ragged_or_nan(a):
        if np.isscalar(a) and np.isnan(a):
            return a
        else:
            return _RaggedElement(a)

    @staticmethod
    def array_or_nan(a):
        if np.isscalar(a) and np.isnan(a):
            return a
        else:
            return a.array

    def __init__(self, array):
        self.array = array

    def __hash__(self):
        return hash(self.array.tobytes())

    def __eq__(self, other):
        if not isinstance(other, _RaggedElement):
            return False
        return np.array_equal(self.array, other.array)

    def __lt__(self, other):
        if not isinstance(other, _RaggedElement):
            return NotImplemented
        return _lexograph_lt(self.array, other.array)

    def __repr__(self):
        array_repr = repr(self.array)
        return array_repr.replace('array', 'ragged_element')
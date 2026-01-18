import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def _matmul_array_vals():
    for a in _array_vals():
        yield a
    for d in _all_dtypes:
        yield ones((3, 4), dtype=d)
        yield ones((4, 2), dtype=d)
        yield ones((4, 4), dtype=d)
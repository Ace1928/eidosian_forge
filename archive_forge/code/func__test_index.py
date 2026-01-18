import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def _test_index(i):
    assert_equal(type(a[i]), mvoid)
    assert_equal_records(a[i]._data, a._data[i])
    assert_equal_records(a[i]._mask, a._mask[i])
    assert_equal(type(a[i, ...]), MaskedArray)
    assert_equal_records(a[i, ...]._data, a._data[i, ...])
    assert_equal_records(a[i, ...]._mask, a._mask[i, ...])
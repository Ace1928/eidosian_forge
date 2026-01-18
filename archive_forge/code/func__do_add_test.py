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
def _do_add_test(self, add):
    assert_(add(np.ma.masked, 1) is np.ma.masked)
    vector = np.array([1, 2, 3])
    result = add(np.ma.masked, vector)
    assert_(result is not np.ma.masked)
    assert_(not isinstance(result, np.ma.core.MaskedConstant))
    assert_equal(result.shape, vector.shape)
    assert_equal(np.ma.getmask(result), np.ones(vector.shape, dtype=bool))
import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def _isin_slow(a, b):
    b = np.asarray(b).flatten().tolist()
    return a in b
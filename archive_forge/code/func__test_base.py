import pytest
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
from numpy.ma.core import MaskedArrayFutureWarning
import io
import textwrap
def _test_base(self, argsort, cls):
    arr_0d = np.array(1).view(cls)
    argsort(arr_0d)
    arr_1d = np.array([1, 2, 3]).view(cls)
    argsort(arr_1d)
    arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
    result = assert_warns(np.ma.core.MaskedArrayFutureWarning, argsort, arr_2d)
    assert_equal(result, argsort(arr_2d, axis=None))
    argsort(arr_2d, axis=None)
    argsort(arr_2d, axis=-1)
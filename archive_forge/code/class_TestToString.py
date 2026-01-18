import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestToString(_DeprecationTestCase):
    message = re.escape('tostring() is deprecated. Use tobytes() instead.')

    def test_tostring(self):
        arr = np.array(list(b'test\xff'), dtype=np.uint8)
        self.assert_deprecated(arr.tostring)

    def test_tostring_matches_tobytes(self):
        arr = np.array(list(b'test\xff'), dtype=np.uint8)
        b = arr.tobytes()
        with assert_warns(DeprecationWarning):
            s = arr.tostring()
        assert s == b
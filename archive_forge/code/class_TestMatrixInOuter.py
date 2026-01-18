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
class TestMatrixInOuter(_DeprecationTestCase):
    message = 'add.outer\\(\\) was passed a numpy matrix as (first|second) argument.'

    def test_deprecated(self):
        arr = np.array([1, 2, 3])
        m = np.array([1, 2, 3]).view(np.matrix)
        self.assert_deprecated(np.add.outer, args=(m, m), num=2)
        self.assert_deprecated(np.add.outer, args=(arr, m))
        self.assert_deprecated(np.add.outer, args=(m, arr))
        self.assert_not_deprecated(np.add.outer, args=(arr, arr))
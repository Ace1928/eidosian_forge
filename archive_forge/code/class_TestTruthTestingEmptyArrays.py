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
class TestTruthTestingEmptyArrays(_DeprecationTestCase):
    message = '.*truth value of an empty array is ambiguous.*'

    def test_1d(self):
        self.assert_deprecated(bool, args=(np.array([]),))

    def test_2d(self):
        self.assert_deprecated(bool, args=(np.zeros((1, 0)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 1)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 0)),))
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
class BuiltInRoundComplexDType(_DeprecationTestCase):
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    not_deprecated_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]

    def test_deprecated(self):
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            self.assert_deprecated(round, args=(scalar,))
            self.assert_deprecated(round, args=(scalar, 0))
            self.assert_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})

    def test_not_deprecated(self):
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            self.assert_not_deprecated(round, args=(scalar,))
            self.assert_not_deprecated(round, args=(scalar, 0))
            self.assert_not_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})
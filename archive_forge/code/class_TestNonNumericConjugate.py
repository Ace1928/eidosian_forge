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
class TestNonNumericConjugate(_DeprecationTestCase):
    """
    Deprecate no-op behavior of ndarray.conjugate on non-numeric dtypes,
    which conflicts with the error behavior of np.conjugate.
    """

    def test_conjugate(self):
        for a in (np.array(5), np.array(5j)):
            self.assert_not_deprecated(a.conjugate)
        for a in (np.array('s'), np.array('2016', 'M'), np.array((1, 2), [('a', int), ('b', int)])):
            self.assert_deprecated(a.conjugate)
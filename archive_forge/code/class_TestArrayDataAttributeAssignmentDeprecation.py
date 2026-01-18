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
class TestArrayDataAttributeAssignmentDeprecation(_DeprecationTestCase):
    """Assigning the 'data' attribute of an ndarray is unsafe as pointed
     out in gh-7093. Eventually, such assignment should NOT be allowed, but
     in the interests of maintaining backwards compatibility, only a Deprecation-
     Warning will be raised instead for the time being to give developers time to
     refactor relevant code.
    """

    def test_data_attr_assignment(self):
        a = np.arange(10)
        b = np.linspace(0, 1, 10)
        self.message = "Assigning the 'data' attribute is an inherently unsafe operation and will be removed in the future."
        self.assert_deprecated(a.__setattr__, args=('data', b.data))
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
class TestDeprecatedUnpickleObjectScalar(_DeprecationTestCase):
    """
    Technically, it should be impossible to create numpy object scalars,
    but there was an unpickle path that would in theory allow it. That
    path is invalid and must lead to the warning.
    """
    message = 'Unpickling a scalar with object dtype is deprecated.'

    def test_deprecated(self):
        ctor = np.core.multiarray.scalar
        self.assert_deprecated(lambda: ctor(np.dtype('O'), 1))
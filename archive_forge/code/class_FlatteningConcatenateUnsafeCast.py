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
class FlatteningConcatenateUnsafeCast(_DeprecationTestCase):
    message = 'concatenate with `axis=None` will use same-kind casting'

    def test_deprecated(self):
        self.assert_deprecated(np.concatenate, args=(([0.0], [1.0]),), kwargs=dict(axis=None, out=np.empty(2, dtype=np.int64)))

    def test_not_deprecated(self):
        self.assert_not_deprecated(np.concatenate, args=(([0.0], [1.0]),), kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64), 'casting': 'unsafe'})
        with assert_raises(TypeError):
            np.concatenate(([0.0], [1.0]), out=np.empty(2, dtype=np.int64), casting='same_kind')
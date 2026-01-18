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
class TestQuantileInterpolationDeprecation(_DeprecationTestCase):

    @pytest.mark.parametrize('func', [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_deprecated(self, func):
        self.assert_deprecated(lambda: func([0.0, 1.0], 0.0, interpolation='linear'))
        self.assert_deprecated(lambda: func([0.0, 1.0], 0.0, interpolation='nearest'))

    @pytest.mark.parametrize('func', [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_both_passed(self, func):
        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            with pytest.raises(TypeError):
                func([0.0, 1.0], 0.0, interpolation='nearest', method='nearest')
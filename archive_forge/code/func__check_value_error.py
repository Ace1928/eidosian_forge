import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def _check_value_error(self, val):
    pattern = '\\(got {}\\)'.format(re.escape(repr(val)))
    with pytest.raises(ValueError, match=pattern) as exc:
        self.conv(val)
import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
class TestArrayToTimedelta64:

    def test_array_to_timedelta64_string_with_unit_2d_raises(self):
        values = np.array([['1', 2], [3, '4']], dtype=object)
        with pytest.raises(ValueError, match='unit must not be specified'):
            array_to_timedelta64(values, unit='s')

    def test_array_to_timedelta64_non_object_raises(self):
        values = np.arange(5)
        msg = "'values' must have object dtype"
        with pytest.raises(TypeError, match=msg):
            array_to_timedelta64(values)
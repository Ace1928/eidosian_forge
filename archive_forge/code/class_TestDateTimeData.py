import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
class TestDateTimeData:

    def test_basic(self):
        a = np.array(['1980-03-23'], dtype=np.datetime64)
        assert_equal(np.datetime_data(a.dtype), ('D', 1))

    def test_bytes(self):
        dt = np.datetime64('2000', (b'ms', 5))
        assert np.datetime_data(dt.dtype) == ('ms', 5)
        dt = np.datetime64('2000', b'5ms')
        assert np.datetime_data(dt.dtype) == ('ms', 5)

    def test_non_ascii(self):
        dt = np.datetime64('2000', ('μs', 5))
        assert np.datetime_data(dt.dtype) == ('us', 5)
        dt = np.datetime64('2000', '5μs')
        assert np.datetime_data(dt.dtype) == ('us', 5)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesSetAxis(SharedSetAxisTests):

    @pytest.fixture
    def obj(self):
        ser = Series(np.arange(4), index=[1, 3, 5, 7], dtype='int64')
        return ser
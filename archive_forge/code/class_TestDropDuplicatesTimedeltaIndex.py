import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDropDuplicatesTimedeltaIndex(DropDuplicates):

    @pytest.fixture
    def idx(self, freq_sample):
        return timedelta_range('1 day', periods=10, freq=freq_sample, name='idx')
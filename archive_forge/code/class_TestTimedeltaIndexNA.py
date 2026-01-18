import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestTimedeltaIndexNA(NATests):

    @pytest.fixture
    def index_without_na(self):
        return TimedeltaIndex(['1 days', '2 days'])
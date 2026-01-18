from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
class TestTesting(Base):
    funcs = ['assert_frame_equal', 'assert_series_equal', 'assert_index_equal', 'assert_extension_array_equal']

    def test_testing(self):
        from pandas import testing
        self.check(testing, self.funcs)

    def test_util_in_top_level(self):
        with pytest.raises(AttributeError, match='foo'):
            pd.util.foo
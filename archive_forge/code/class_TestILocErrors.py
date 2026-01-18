from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestILocErrors:

    def test_iloc_float_raises(self, series_with_simple_index, frame_or_series, warn_copy_on_write):
        obj = series_with_simple_index
        if frame_or_series is DataFrame:
            obj = obj.to_frame()
        msg = 'Cannot index by location index with a non-integer key'
        with pytest.raises(TypeError, match=msg):
            obj.iloc[3.0]
        with pytest.raises(IndexError, match=_slice_iloc_msg):
            with tm.assert_cow_warning(warn_copy_on_write and frame_or_series is DataFrame):
                obj.iloc[3.0] = 0

    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame):
        with pytest.raises(IndexingError, match='Too many indexers'):
            float_frame.iloc[:, :, :]
        with pytest.raises(IndexError, match='too many indices for array'):
            float_frame.iloc[:, :, :] = 1

    def test_iloc_frame_indexer(self):
        df = DataFrame({'a': [1, 2, 3]})
        indexer = DataFrame({'a': [True, False, True]})
        msg = 'DataFrame indexer for .iloc is not supported. Consider using .loc'
        with pytest.raises(TypeError, match=msg):
            df.iloc[indexer] = 1
        msg = 'DataFrame indexer is not allowed for .iloc\nConsider using .loc for automatic alignment.'
        with pytest.raises(IndexError, match=msg):
            df.iloc[indexer]
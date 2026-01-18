import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
class TestToDictOfBlocks:

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_no_copy_blocks(self, float_frame, using_copy_on_write):
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]
        _last_df = None
        blocks = df._to_dict_of_blocks()
        for _df in blocks.values():
            _last_df = _df
            if column in _df:
                _df.loc[:, column] = _df[column] + 1
        if not using_copy_on_write:
            assert _last_df is not None and _last_df[column].equals(df[column])
        else:
            assert _last_df is not None and (not _last_df[column].equals(df[column]))
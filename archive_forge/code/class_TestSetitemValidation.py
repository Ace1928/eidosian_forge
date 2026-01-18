from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSetitemValidation:

    def _check_setitem_invalid(self, df, invalid, indexer, warn):
        msg = 'Setting an item of incompatible dtype is deprecated'
        msg = re.escape(msg)
        orig_df = df.copy()
        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[indexer, 0] = invalid
            df = orig_df.copy()
        with tm.assert_produces_warning(warn, match=msg):
            df.loc[indexer, 'a'] = invalid
            df = orig_df.copy()
    _invalid_scalars = [1 + 2j, 'True', '1', '1.0', pd.NaT, np.datetime64('NaT'), np.timedelta64('NaT')]
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize('invalid', _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)])
    @pytest.mark.parametrize('indexer', _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        df = DataFrame({'a': [True, False, False]}, dtype='bool')
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)

    @pytest.mark.parametrize('invalid', _invalid_scalars + [True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize('indexer', _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        df = DataFrame({'a': [1, 2, 3]}, dtype=any_int_numpy_dtype)
        if isna(invalid) and invalid is not pd.NaT and (not np.isnat(invalid)):
            warn = None
        else:
            warn = FutureWarning
        self._check_setitem_invalid(df, invalid, indexer, warn)

    @pytest.mark.parametrize('invalid', _invalid_scalars + [True])
    @pytest.mark.parametrize('indexer', _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        df = DataFrame({'a': [1, 2, None]}, dtype=float_numpy_dtype)
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)
from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
class TestAstypeString:

    @pytest.mark.parametrize('data, dtype', [([True, NA], 'boolean'), (['A', NA], 'category'), (['2020-10-10', '2020-10-10'], 'datetime64[ns]'), (['2020-10-10', '2020-10-10', NaT], 'datetime64[ns]'), (['2012-01-01 00:00:00-05:00', NaT], 'datetime64[ns, US/Eastern]'), ([1, None], 'UInt16'), (['1/1/2021', '2/1/2021'], 'period[M]'), (['1/1/2021', '2/1/2021', NaT], 'period[M]'), (['1 Day', '59 Days', NaT], 'timedelta64[ns]')])
    def test_astype_string_to_extension_dtype_roundtrip(self, data, dtype, request, nullable_string_dtype):
        if dtype == 'boolean':
            mark = pytest.mark.xfail(reason='TODO StringArray.astype() with missing values #GH40566')
            request.applymarker(mark)
        ser = Series(data, dtype=dtype)
        result = ser.astype(nullable_string_dtype).astype(ser.dtype)
        tm.assert_series_equal(result, ser)
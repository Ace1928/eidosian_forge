from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocCallable:

    def test_frame_loc_getitem_callable(self):
        df = DataFrame({'A': [1, 2, 3, 4], 'B': list('aabb'), 'C': [1, 2, 3, 4]})
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])
        res = df.loc[lambda x: x.B == 'b', :]
        tm.assert_frame_equal(res, df.loc[df.B == 'b', :])
        res = df.loc[lambda x: x.A > 2, lambda x: x.columns == 'B']
        tm.assert_frame_equal(res, df.loc[df.A > 2, [False, True, False]])
        res = df.loc[lambda x: x.A > 2, lambda x: 'B']
        tm.assert_series_equal(res, df.loc[df.A > 2, 'B'])
        res = df.loc[lambda x: x.A > 2, lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
        res = df.loc[lambda x: x.A == 2, lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ['A', 'B']])
        res = df.loc[lambda x: 1, lambda x: 'A']
        assert res == df.loc[1, 'A']

    def test_frame_loc_getitem_callable_mixture(self):
        df = DataFrame({'A': [1, 2, 3, 4], 'B': list('aabb'), 'C': [1, 2, 3, 4]})
        res = df.loc[lambda x: x.A > 2, ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
        res = df.loc[[2, 3], lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[[2, 3], ['A', 'B']])
        res = df.loc[3, lambda x: ['A', 'B']]
        tm.assert_series_equal(res, df.loc[3, ['A', 'B']])

    def test_frame_loc_getitem_callable_labels(self):
        df = DataFrame({'X': [1, 2, 3, 4], 'Y': list('aabb')}, index=list('ABCD'))
        res = df.loc[lambda x: ['A', 'C']]
        tm.assert_frame_equal(res, df.loc[['A', 'C']])
        res = df.loc[lambda x: ['A', 'C'], :]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], :])
        res = df.loc[lambda x: ['A', 'C'], lambda x: 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[lambda x: ['A', 'C'], lambda x: ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])
        res = df.loc[['A', 'C'], lambda x: 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[['A', 'C'], lambda x: ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])
        res = df.loc[lambda x: ['A', 'C'], 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[lambda x: ['A', 'C'], ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])

    def test_frame_loc_setitem_callable(self):
        df = DataFrame({'X': [1, 2, 3, 4], 'Y': Series(list('aabb'), dtype=object)}, index=list('ABCD'))
        res = df.copy()
        res.loc[lambda x: ['A', 'C']] = -20
        exp = df.copy()
        exp.loc[['A', 'C']] = -20
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[lambda x: ['A', 'C'], :] = 20
        exp = df.copy()
        exp.loc[['A', 'C'], :] = 20
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[lambda x: ['A', 'C'], lambda x: 'X'] = -1
        exp = df.copy()
        exp.loc[['A', 'C'], 'X'] = -1
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[lambda x: ['A', 'C'], lambda x: ['X']] = [5, 10]
        exp = df.copy()
        exp.loc[['A', 'C'], ['X']] = [5, 10]
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[['A', 'C'], lambda x: 'X'] = np.array([-1, -2])
        exp = df.copy()
        exp.loc[['A', 'C'], 'X'] = np.array([-1, -2])
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[['A', 'C'], lambda x: ['X']] = 10
        exp = df.copy()
        exp.loc[['A', 'C'], ['X']] = 10
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[lambda x: ['A', 'C'], 'X'] = -2
        exp = df.copy()
        exp.loc[['A', 'C'], 'X'] = -2
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.loc[lambda x: ['A', 'C'], ['X']] = -4
        exp = df.copy()
        exp.loc[['A', 'C'], ['X']] = -4
        tm.assert_frame_equal(res, exp)
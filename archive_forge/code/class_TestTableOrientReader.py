from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
class TestTableOrientReader:

    @pytest.mark.parametrize('index_nm', [None, 'idx', pytest.param('index', marks=pytest.mark.xfail), 'level_0'])
    @pytest.mark.parametrize('vals', [{'ints': [1, 2, 3, 4]}, {'objects': ['a', 'b', 'c', 'd']}, {'objects': ['1', '2', '3', '4']}, {'date_ranges': pd.date_range('2016-01-01', freq='d', periods=4)}, {'categoricals': pd.Series(pd.Categorical(['a', 'b', 'c', 'c']))}, {'ordered_cats': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True))}, {'floats': [1.0, 2.0, 3.0, 4.0]}, {'floats': [1.1, 2.2, 3.3, 4.4]}, {'bools': [True, False, False, True]}, {'timezones': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}])
    def test_read_json_table_orient(self, index_nm, vals, recwarn):
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('index_nm', [None, 'idx', 'index'])
    @pytest.mark.parametrize('vals', [{'timedeltas': pd.timedelta_range('1h', periods=4, freq='min')}])
    def test_read_json_table_orient_raises(self, index_nm, vals, recwarn):
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient='table')
        with pytest.raises(NotImplementedError, match='can not yet read '):
            pd.read_json(out, orient='table')

    @pytest.mark.parametrize('index_nm', [None, 'idx', pytest.param('index', marks=pytest.mark.xfail), 'level_0'])
    @pytest.mark.parametrize('vals', [{'ints': [1, 2, 3, 4]}, {'objects': ['a', 'b', 'c', 'd']}, {'objects': ['1', '2', '3', '4']}, {'date_ranges': pd.date_range('2016-01-01', freq='d', periods=4)}, {'categoricals': pd.Series(pd.Categorical(['a', 'b', 'c', 'c']))}, {'ordered_cats': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True))}, {'floats': [1.0, 2.0, 3.0, 4.0]}, {'floats': [1.1, 2.2, 3.3, 4.4]}, {'bools': [True, False, False, True]}, {'timezones': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}])
    def test_read_json_table_period_orient(self, index_nm, vals, recwarn):
        df = DataFrame(vals, index=pd.Index((pd.Period(f'2022Q{q}') for q in range(1, 5)), name=index_nm))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('idx', [pd.Index(range(4)), pd.date_range('2020-08-30', freq='d', periods=4)._with_freq(None), pd.date_range('2020-08-30', freq='d', periods=4, tz='US/Central')._with_freq(None), pd.MultiIndex.from_product([pd.date_range('2020-08-30', freq='d', periods=2, tz='US/Central'), ['x', 'y']])])
    @pytest.mark.parametrize('vals', [{'floats': [1.1, 2.2, 3.3, 4.4]}, {'dates': pd.date_range('2020-08-30', freq='d', periods=4)}, {'timezones': pd.date_range('2020-08-30', freq='d', periods=4, tz='Europe/London')}])
    def test_read_json_table_timezones_orient(self, idx, vals, recwarn):
        df = DataFrame(vals, index=idx)
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    def test_comprehensive(self):
        df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'c'], 'C': pd.date_range('2016-01-01', freq='d', periods=4), 'E': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'])), 'F': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True)), 'G': [1.1, 2.2, 3.3, 4.4], 'H': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central'), 'I': [True, False, False, True]}, index=pd.Index(range(4), name='idx'))
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('index_names', [[None, None], ['foo', 'bar'], ['foo', None], [None, 'foo'], ['index', 'foo']])
    def test_multiindex(self, index_names):
        df = DataFrame([['Arr', 'alpha', [1, 2, 3, 4]], ['Bee', 'Beta', [10, 20, 30, 40]]], index=[['A', 'B'], ['Null', 'Eins']], columns=['Aussprache', 'Griechisch', 'Args'])
        df.index.names = index_names
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)

    def test_empty_frame_roundtrip(self):
        df = DataFrame(columns=['a', 'b', 'c'])
        expected = df.copy()
        out = StringIO(df.to_json(orient='table'))
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(expected, result)

    def test_read_json_orient_table_old_schema_version(self):
        df_json = '\n        {\n            "schema":{\n                "fields":[\n                    {"name":"index","type":"integer"},\n                    {"name":"a","type":"string"}\n                ],\n                "primaryKey":["index"],\n                "pandas_version":"0.20.0"\n            },\n            "data":[\n                {"index":0,"a":1},\n                {"index":1,"a":2.0},\n                {"index":2,"a":"s"}\n            ]\n        }\n        '
        expected = DataFrame({'a': [1, 2.0, 's']})
        result = pd.read_json(StringIO(df_json), orient='table')
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize('freq', ['M', '2M', 'Q', '2Q', 'Y', '2Y'])
    def test_read_json_table_orient_period_depr_freq(self, freq, recwarn):
        df = DataFrame({'ints': [1, 2]}, index=pd.PeriodIndex(['2020-01', '2021-06'], freq=freq))
        out = df.to_json(orient='table')
        result = pd.read_json(out, orient='table')
        tm.assert_frame_equal(df, result)
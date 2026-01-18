import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
class TestPreserves:

    @pytest.mark.parametrize('cls, data', [(pd.Series, np.array([])), (pd.Series, [1, 2]), (pd.DataFrame, {}), (pd.DataFrame, {'A': [1, 2]})])
    def test_construction_ok(self, cls, data):
        result = cls(data)
        assert result.flags.allows_duplicate_labels is True
        result = cls(data).set_flags(allows_duplicate_labels=False)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('func', [operator.itemgetter(['a']), operator.methodcaller('add', 1), operator.methodcaller('rename', str.upper), operator.methodcaller('rename', 'name'), operator.methodcaller('abs'), np.abs])
    def test_preserved_series(self, func):
        s = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        assert func(s).flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('other', [pd.Series(0, index=['a', 'b', 'c']), pd.Series(0, index=['a', 'b'])])
    @not_implemented
    def test_align(self, other):
        s = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        a, b = s.align(other)
        assert a.flags.allows_duplicate_labels is False
        assert b.flags.allows_duplicate_labels is False

    def test_preserved_frame(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        assert df.loc[['a']].flags.allows_duplicate_labels is False
        assert df.loc[:, ['A', 'B']].flags.allows_duplicate_labels is False

    def test_to_frame(self):
        ser = pd.Series(dtype=float).set_flags(allows_duplicate_labels=False)
        assert ser.to_frame().flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('func', ['add', 'sub'])
    @pytest.mark.parametrize('frame', [False, True])
    @pytest.mark.parametrize('other', [1, pd.Series([1, 2], name='A')])
    def test_binops(self, func, other, frame):
        df = pd.Series([1, 2], name='A', index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        if frame:
            df = df.to_frame()
        if isinstance(other, pd.Series) and frame:
            other = other.to_frame()
        func = operator.methodcaller(func, other)
        assert df.flags.allows_duplicate_labels is False
        assert func(df).flags.allows_duplicate_labels is False

    def test_preserve_getitem(self):
        df = pd.DataFrame({'A': [1, 2]}).set_flags(allows_duplicate_labels=False)
        assert df[['A']].flags.allows_duplicate_labels is False
        assert df['A'].flags.allows_duplicate_labels is False
        assert df.loc[0].flags.allows_duplicate_labels is False
        assert df.loc[[0]].flags.allows_duplicate_labels is False
        assert df.loc[0, ['A']].flags.allows_duplicate_labels is False

    def test_ndframe_getitem_caching_issue(self, request, using_copy_on_write, warn_copy_on_write):
        if not (using_copy_on_write or warn_copy_on_write):
            request.applymarker(pytest.mark.xfail(reason='Unclear behavior.'))
        df = pd.DataFrame({'A': [0]}).set_flags(allows_duplicate_labels=False)
        assert df['A'].flags.allows_duplicate_labels is False
        df.flags.allows_duplicate_labels = True
        assert df['A'].flags.allows_duplicate_labels is True

    @pytest.mark.parametrize('objs, kwargs', [([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['c', 'd'])], {}), ([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['a', 'b'])], {'ignore_index': True}), ([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['a', 'b'])], {'axis': 1}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'A': [1, 2]}, index=['c', 'd'])], {}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])], {'ignore_index': True}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'B': [1, 2]}, index=['a', 'b'])], {'axis': 1}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.Series([1, 2], index=['a', 'b'], name='B')], {'axis': 1})])
    def test_concat(self, objs, kwargs):
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        result = pd.concat(objs, **kwargs)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('left, right, expected', [pytest.param(pd.DataFrame({'A': [0, 1]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']).set_flags(allows_duplicate_labels=False), False, marks=not_implemented), pytest.param(pd.DataFrame({'A': [0, 1]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']), False, marks=not_implemented), (pd.DataFrame({'A': [0, 1]}, index=['a', 'b']), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']), True)])
    def test_merge(self, left, right, expected):
        result = pd.merge(left, right, left_index=True, right_index=True)
        assert result.flags.allows_duplicate_labels is expected

    @not_implemented
    def test_groupby(self):
        df = pd.DataFrame({'A': [1, 2, 3]}).set_flags(allows_duplicate_labels=False)
        result = df.groupby([0, 0, 1]).agg('count')
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('frame', [True, False])
    @not_implemented
    def test_window(self, frame):
        df = pd.Series(1, index=pd.date_range('2000', periods=12), name='A', allows_duplicate_labels=False)
        if frame:
            df = df.to_frame()
        assert df.rolling(3).mean().flags.allows_duplicate_labels is False
        assert df.ewm(3).mean().flags.allows_duplicate_labels is False
        assert df.expanding(3).mean().flags.allows_duplicate_labels is False
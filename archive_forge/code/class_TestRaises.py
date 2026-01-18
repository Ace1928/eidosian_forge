import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
class TestRaises:

    @pytest.mark.parametrize('cls, axes', [(pd.Series, {'index': ['a', 'a'], 'dtype': float}), (pd.DataFrame, {'index': ['a', 'a']}), (pd.DataFrame, {'index': ['a', 'a'], 'columns': ['b', 'b']}), (pd.DataFrame, {'columns': ['b', 'b']})])
    def test_set_flags_with_duplicates(self, cls, axes):
        result = cls(**axes)
        assert result.flags.allows_duplicate_labels is True
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            cls(**axes).set_flags(allows_duplicate_labels=False)

    @pytest.mark.parametrize('data', [pd.Series(index=[0, 0], dtype=float), pd.DataFrame(index=[0, 0]), pd.DataFrame(columns=[0, 0])])
    def test_setting_allows_duplicate_labels_raises(self, data):
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            data.flags.allows_duplicate_labels = False
        assert data.flags.allows_duplicate_labels is True

    def test_series_raises(self):
        a = pd.Series(0, index=['a', 'b'])
        b = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat([a, b])

    @pytest.mark.parametrize('getter, target', [(operator.itemgetter(['A', 'A']), None), (operator.itemgetter(['a', 'a']), 'loc'), pytest.param(operator.itemgetter(('a', ['A', 'A'])), 'loc'), (operator.itemgetter((['a', 'a'], 'A')), 'loc'), (operator.itemgetter([0, 0]), 'iloc'), pytest.param(operator.itemgetter((0, [0, 0])), 'iloc'), pytest.param(operator.itemgetter(([0, 0], 0)), 'iloc')])
    def test_getitem_raises(self, getter, target):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        if target:
            target = getattr(df, target)
        else:
            target = df
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            getter(target)

    @pytest.mark.parametrize('objs, kwargs', [([pd.Series(1, index=[0, 1], name='a'), pd.Series(2, index=[0, 1], name='a')], {'axis': 1})])
    def test_concat_raises(self, objs, kwargs):
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat(objs, **kwargs)

    @not_implemented
    def test_merge_raises(self):
        a = pd.DataFrame({'A': [0, 1, 2]}, index=['a', 'b', 'c']).set_flags(allows_duplicate_labels=False)
        b = pd.DataFrame({'B': [0, 1, 2]}, index=['a', 'b', 'b'])
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.merge(a, b, left_index=True, right_index=True)
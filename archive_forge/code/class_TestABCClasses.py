import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
class TestABCClasses:
    tuples = [[1, 2, 2], ['red', 'blue', 'red']]
    multi_index = pd.MultiIndex.from_arrays(tuples, names=('number', 'color'))
    datetime_index = pd.to_datetime(['2000/1/1', '2010/1/1'])
    timedelta_index = pd.to_timedelta(np.arange(5), unit='s')
    period_index = pd.period_range('2000/1/1', '2010/1/1/', freq='M')
    categorical = pd.Categorical([1, 2, 3], categories=[2, 3, 1])
    categorical_df = pd.DataFrame({'values': [1, 2, 3]}, index=categorical)
    df = pd.DataFrame({'names': ['a', 'b', 'c']}, index=multi_index)
    sparse_array = pd.arrays.SparseArray(np.random.default_rng(2).standard_normal(10))
    datetime_array = pd.core.arrays.DatetimeArray._from_sequence(datetime_index)
    timedelta_array = pd.core.arrays.TimedeltaArray._from_sequence(timedelta_index)
    abc_pairs = [('ABCMultiIndex', multi_index), ('ABCDatetimeIndex', datetime_index), ('ABCRangeIndex', pd.RangeIndex(3)), ('ABCTimedeltaIndex', timedelta_index), ('ABCIntervalIndex', pd.interval_range(start=0, end=3)), ('ABCPeriodArray', pd.arrays.PeriodArray([2000, 2001, 2002], dtype='period[D]')), ('ABCNumpyExtensionArray', pd.arrays.NumpyExtensionArray(np.array([0, 1, 2]))), ('ABCPeriodIndex', period_index), ('ABCCategoricalIndex', categorical_df.index), ('ABCSeries', pd.Series([1, 2, 3])), ('ABCDataFrame', df), ('ABCCategorical', categorical), ('ABCDatetimeArray', datetime_array), ('ABCTimedeltaArray', timedelta_array)]

    @pytest.mark.parametrize('abctype1, inst', abc_pairs)
    @pytest.mark.parametrize('abctype2, _', abc_pairs)
    def test_abc_pairs_instance_check(self, abctype1, abctype2, inst, _):
        if abctype1 == abctype2:
            assert isinstance(inst, getattr(gt, abctype2))
            assert not isinstance(type(inst), getattr(gt, abctype2))
        else:
            assert not isinstance(inst, getattr(gt, abctype2))

    @pytest.mark.parametrize('abctype1, inst', abc_pairs)
    @pytest.mark.parametrize('abctype2, _', abc_pairs)
    def test_abc_pairs_subclass_check(self, abctype1, abctype2, inst, _):
        if abctype1 == abctype2:
            assert issubclass(type(inst), getattr(gt, abctype2))
            with pytest.raises(TypeError, match=re.escape('issubclass() arg 1 must be a class')):
                issubclass(inst, getattr(gt, abctype2))
        else:
            assert not issubclass(type(inst), getattr(gt, abctype2))
    abc_subclasses = {'ABCIndex': [abctype for abctype, _ in abc_pairs if 'Index' in abctype and abctype != 'ABCIndex'], 'ABCNDFrame': ['ABCSeries', 'ABCDataFrame'], 'ABCExtensionArray': ['ABCCategorical', 'ABCDatetimeArray', 'ABCPeriodArray', 'ABCTimedeltaArray']}

    @pytest.mark.parametrize('parent, subs', abc_subclasses.items())
    @pytest.mark.parametrize('abctype, inst', abc_pairs)
    def test_abc_hierarchy(self, parent, subs, abctype, inst):
        if abctype in subs:
            assert isinstance(inst, getattr(gt, parent))
        else:
            assert not isinstance(inst, getattr(gt, parent))

    @pytest.mark.parametrize('abctype', [e for e in gt.__dict__ if e.startswith('ABC')])
    def test_abc_coverage(self, abctype):
        assert abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
class TestSampleDataFrame:

    def test_sample(self):
        easy_weight_list = [0] * 10
        easy_weight_list[5] = 1
        df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': ['a'] * 10, 'easyweights': easy_weight_list})
        sample1 = df.sample(n=1, weights='easyweights')
        tm.assert_frame_equal(sample1, df.iloc[5:6])
        ser = Series(range(10))
        msg = 'Strings cannot be passed as weights when sampling from a Series.'
        with pytest.raises(ValueError, match=msg):
            ser.sample(n=3, weights='weight_column')
        msg = 'Strings can only be passed to weights when sampling from rows on a DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, weights='weight_column', axis=1)
        with pytest.raises(KeyError, match="'String passed to weights not a valid column'"):
            df.sample(n=3, weights='not_a_real_column_name')
        weights_less_than_1 = [0] * 10
        weights_less_than_1[0] = 0.5
        tm.assert_frame_equal(df.sample(n=1, weights=weights_less_than_1), df.iloc[:1])
        df = DataFrame({'col1': range(10), 'col2': ['a'] * 10})
        second_column_weight = [0, 1]
        tm.assert_frame_equal(df.sample(n=1, axis=1, weights=second_column_weight), df[['col2']])
        tm.assert_frame_equal(df.sample(n=1, axis='columns', weights=second_column_weight), df[['col2']])
        weight = [0] * 10
        weight[5] = 0.5
        tm.assert_frame_equal(df.sample(n=1, axis='rows', weights=weight), df.iloc[5:6])
        tm.assert_frame_equal(df.sample(n=1, axis='index', weights=weight), df.iloc[5:6])
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=2)
        msg = 'No axis named not_a_name for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis='not_a_name')
        ser = Series(range(10))
        with pytest.raises(ValueError, match='No axis named 1 for object type Series'):
            ser.sample(n=1, axis=1)
        msg = 'Weights and axis to be sampled must be of same length'
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=1, weights=[0.5] * 10)

    def test_sample_axis1(self):
        easy_weight_list = [0] * 3
        easy_weight_list[2] = 1
        df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': ['a'] * 10})
        sample1 = df.sample(n=1, axis=1, weights=easy_weight_list)
        tm.assert_frame_equal(sample1, df[['colString']])
        tm.assert_frame_equal(df.sample(n=3, random_state=42), df.sample(n=3, axis=0, random_state=42))

    def test_sample_aligns_weights_with_frame(self):
        df = DataFrame({'col1': [5, 6, 7], 'col2': ['a', 'b', 'c']}, index=[9, 5, 3])
        ser = Series([1, 0, 0], index=[3, 5, 9])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser))
        ser2 = Series([0.001, 0, 10000], index=[3, 5, 10])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser2))
        ser3 = Series([0.01, 0], index=[3, 5])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser3))
        ser4 = Series([1, 0], index=[1, 2])
        with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
            df.sample(1, weights=ser4)

    def test_sample_is_copy(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        df2 = df.sample(3)
        with tm.assert_produces_warning(None):
            df2['d'] = 1

    def test_sample_does_not_modify_weights(self):
        result = np.array([np.nan, 1, np.nan])
        expected = result.copy()
        ser = Series([1, 2, 3])
        ser.sample(weights=result)
        tm.assert_numpy_array_equal(result, expected)
        df = DataFrame({'values': [1, 1, 1], 'weights': [1, np.nan, np.nan]})
        expected = df['weights'].copy()
        df.sample(frac=1.0, replace=True, weights='weights')
        result = df['weights']
        tm.assert_series_equal(result, expected)

    def test_sample_ignore_index(self):
        df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': ['a'] * 10})
        result = df.sample(3, ignore_index=True)
        expected_index = Index(range(3))
        tm.assert_index_equal(result.index, expected_index, exact=True)
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
class CheckGrouping:

    @pytest.mark.smoke
    def test_reindex(self):
        self.grouping.reindex(self.grouping.index)

    def test_count_categories(self):
        self.grouping.count_categories(level=0)
        np.testing.assert_equal(self.grouping.counts, self.expected_counts)

    def test_sort(self):
        sorted_data, index = self.grouping.sort(self.data)
        expected_sorted_data = self.data.sort_index()
        assert_frame_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.DataFrame))
        np.testing.assert_(not index.equals(self.grouping.index))
        if hasattr(sorted_data, 'equals'):
            np.testing.assert_(not sorted_data.equals(self.data))
        sorted_data, index = self.grouping.sort(self.data.values)
        np.testing.assert_array_equal(sorted_data, expected_sorted_data.values)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))
        series = self.data[self.data.columns[0]]
        sorted_data, index = self.grouping.sort(series)
        expected_sorted_data = series.sort_index()
        assert_series_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.Series))
        if hasattr(sorted_data, 'equals'):
            np.testing.assert_(not sorted_data.equals(series))
        array = series.values
        sorted_data, index = self.grouping.sort(array)
        expected_sorted_data = series.sort_index().values
        np.testing.assert_array_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))

    def test_transform_dataframe(self):
        names = self.data.index.names
        transformed_dataframe = self.grouping.transform_dataframe(self.data, lambda x: x.mean(), level=0)
        cols = [names[0]] + list(self.data.columns)
        df = self.data.reset_index()[cols].set_index(names[0])
        grouped = df[self.data.columns].groupby(level=0)
        expected = grouped.apply(lambda x: x.mean())
        np.testing.assert_allclose(transformed_dataframe, expected.values)
        if len(names) > 1:
            transformed_dataframe = self.grouping.transform_dataframe(self.data, lambda x: x.mean(), level=1)
            cols = [names[1]] + list(self.data.columns)
            df = self.data.reset_index()[cols].set_index(names[1])
            grouped = df.groupby(level=0)
            expected = grouped.apply(lambda x: x.mean())[self.data.columns]
            np.testing.assert_allclose(transformed_dataframe, expected.values)

    def test_transform_array(self):
        names = self.data.index.names
        transformed_array = self.grouping.transform_array(self.data.values, lambda x: x.mean(), level=0)
        cols = [names[0]] + list(self.data.columns)
        df = self.data.reset_index()[cols].set_index(names[0])
        grouped = df[self.data.columns].groupby(level=0)
        expected = grouped.apply(lambda x: x.mean())
        np.testing.assert_allclose(transformed_array, expected.values)
        if len(names) > 1:
            transformed_array = self.grouping.transform_array(self.data.values, lambda x: x.mean(), level=1)
            cols = [names[1]] + list(self.data.columns)
            df = self.data.reset_index()[cols].set_index(names[1])
            grouped = df[self.data.columns].groupby(level=0)
            expected = grouped.apply(lambda x: x.mean())[self.data.columns]
            np.testing.assert_allclose(transformed_array, expected.values)

    def test_transform_slices(self):
        names = self.data.index.names
        transformed_slices = self.grouping.transform_slices(self.data.values, lambda x, idx: x.mean(0), level=0)
        expected = self.data.reset_index().groupby(names[0])[self.data.columns].mean()
        np.testing.assert_allclose(transformed_slices, expected.values, rtol=1e-12, atol=1e-25)
        if len(names) > 1:
            transformed_slices = self.grouping.transform_slices(self.data.values, lambda x, idx: x.mean(0), level=1)
            expected = self.data.reset_index().groupby(names[1])[self.data.columns].mean()
            np.testing.assert_allclose(transformed_slices, expected.values, rtol=1e-12, atol=1e-25)

    @pytest.mark.smoke
    def test_dummies_groups(self):
        self.grouping.dummies_groups()
        if len(self.grouping.group_names) > 1:
            self.grouping.dummies_groups(level=1)

    def test_dummy_sparse(self):
        data = self.data
        self.grouping.dummy_sparse()
        values = data.index.get_level_values(0).values
        expected = pd.get_dummies(pd.Series(values, dtype='category'), drop_first=False)
        np.testing.assert_equal(self.grouping._dummies.toarray(), expected)
        if len(self.grouping.group_names) > 1:
            self.grouping.dummy_sparse(level=1)
            values = data.index.get_level_values(1).values
            expected = pd.get_dummies(pd.Series(values, dtype='category'), drop_first=False)
            np.testing.assert_equal(self.grouping._dummies.toarray(), expected)
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.skipif(get_current_execution() == 'BaseOnPython', reason="BaseOnPython doesn't have proxy categories")
class TestCategoricalProxyDtype:
    """This class contains test and test usilities for the ``LazyProxyCategoricalDtype`` class."""

    @staticmethod
    def _get_lazy_proxy():
        """
        Build a dataframe containing a column that has a proxy type and return
        this proxy together with an original dtype that this proxy is emulating.

        Returns
        -------
        (LazyProxyCategoricalDtype, pandas.CategoricalDtype, modin.pandas.DataFrame)
        """
        nchunks = 3
        pandas_df = pandas.DataFrame({'a': [1, 1, 2, 2, 3, 2], 'b': [1, 2, 3, 4, 5, 6]})
        original_dtype = pandas_df.astype({'a': 'category'}).dtypes['a']
        chunks = split_result_of_axis_func_pandas(axis=0, num_splits=nchunks, result=pandas_df, min_block_size=MinPartitionSize.get(), length_list=[2, 2, 2])
        if StorageFormat.get() == 'Pandas':
            df = pd.concat([pd.DataFrame(chunk) for chunk in chunks])
            assert df._query_compiler._modin_frame._partitions.shape == (nchunks, 1)
            df = df.astype({'a': 'category'})
            return (df.dtypes['a'], original_dtype, df)
        elif StorageFormat.get() == 'Hdk':
            import pyarrow as pa
            from modin.pandas.io import from_arrow
            at = pa.concat_tables([pa.Table.from_pandas(chunk.astype({'a': 'category'})) for chunk in chunks])
            assert len(at.column(0).chunks) == nchunks
            df = from_arrow(at)
            return (df.dtypes['a'], original_dtype, df)
        else:
            raise NotImplementedError()

    def test_update_proxy(self):
        """Verify that ``LazyProxyCategoricalDtype._update_proxy`` method works as expected."""
        lazy_proxy, _, _ = self._get_lazy_proxy()
        new_parent = pd.DataFrame({'a': [10, 20, 30]})._query_compiler._modin_frame
        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        assert lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name) is lazy_proxy
        proxy_with_new_column = lazy_proxy._update_proxy(lazy_proxy._parent, 'other_column')
        assert proxy_with_new_column is not lazy_proxy and isinstance(proxy_with_new_column, LazyProxyCategoricalDtype)
        proxy_with_new_parent = lazy_proxy._update_proxy(new_parent, lazy_proxy._column_name)
        assert proxy_with_new_parent is not lazy_proxy and isinstance(proxy_with_new_parent, LazyProxyCategoricalDtype)
        lazy_proxy.categories
        assert type(lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name)) == pandas.CategoricalDtype

    def test_update_proxy_implicit(self):
        """
        Verify that a lazy proxy correctly updates its parent when passed from one parent to another.
        """
        lazy_proxy, _, parent = self._get_lazy_proxy()
        parent_frame = parent._query_compiler._modin_frame
        if StorageFormat.get() == 'Pandas':
            assert lazy_proxy._parent is parent_frame
        elif StorageFormat.get() == 'Hdk':
            arrow_table = parent_frame._partitions[0, 0].get()
            assert lazy_proxy._parent is arrow_table
        else:
            raise NotImplementedError(f'The test is not implemented for {StorageFormat.get()} storage format')
        new_parent = parent.copy()
        new_parent_frame = new_parent._query_compiler._modin_frame
        new_lazy_proxy = new_parent_frame.dtypes[lazy_proxy._column_name]
        if StorageFormat.get() == 'Pandas':
            assert lazy_proxy._parent is parent_frame
            assert new_lazy_proxy._parent is new_parent_frame
        elif StorageFormat.get() == 'Hdk':
            new_arrow_table = new_parent_frame._partitions[0, 0].get()
            assert lazy_proxy._parent is arrow_table
            assert new_lazy_proxy._parent is new_arrow_table
        else:
            raise NotImplementedError(f'The test is not implemented for {StorageFormat.get()} storage format')

    def test_if_proxy_lazy(self):
        """Verify that proxy is able to pass simple comparison checks without triggering materialization."""
        lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()
        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        assert not lazy_proxy._is_materialized
        assert lazy_proxy == 'category'
        assert isinstance(lazy_proxy, pd.CategoricalDtype)
        assert isinstance(lazy_proxy, pandas.CategoricalDtype)
        assert str(lazy_proxy) == 'category'
        assert str(lazy_proxy) == str(actual_dtype)
        assert not lazy_proxy.ordered
        assert not lazy_proxy._is_materialized
        assert lazy_proxy == actual_dtype
        assert actual_dtype == lazy_proxy
        assert repr(lazy_proxy) == repr(actual_dtype)
        assert lazy_proxy.categories.equals(actual_dtype.categories)
        assert lazy_proxy._is_materialized

    def test_proxy_as_dtype(self):
        """Verify that proxy can be used as an actual dtype."""
        lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()
        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        assert not lazy_proxy._is_materialized
        modin_df2, pandas_df2 = create_test_dfs({'c': [2, 2, 3, 4, 5, 6]})
        eval_general((modin_df2, lazy_proxy), (pandas_df2, actual_dtype), lambda args: args[0].astype({'c': args[1]}))

    def test_proxy_with_pandas_constructor(self):
        """Verify that users still can use pandas' constructor using `type(cat)(...)` notation."""
        lazy_proxy, _, _ = self._get_lazy_proxy()
        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        new_cat_values = pandas.Index([3, 4, 5])
        new_category_dtype = type(lazy_proxy)(categories=new_cat_values, ordered=True)
        assert not lazy_proxy._is_materialized
        assert new_category_dtype._is_materialized
        assert new_category_dtype.categories.equals(new_cat_values)
        assert new_category_dtype.ordered
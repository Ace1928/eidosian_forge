from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestDatasetView:

    def test_view_contents(self):
        ds = create_test_data()
        dt: DataTree = DataTree(data=ds)
        assert ds.identical(dt.ds)
        assert isinstance(dt.ds, xr.Dataset)

    def test_immutability(self):
        dt: DataTree = DataTree(name='root', data=None)
        DataTree(name='a', data=None, parent=dt)
        with pytest.raises(AttributeError, match='Mutation of the DatasetView is not allowed'):
            dt.ds['a'] = xr.DataArray(0)
        with pytest.raises(AttributeError, match='Mutation of the DatasetView is not allowed'):
            dt.ds.update({'a': 0})

    def test_methods(self):
        ds = create_test_data()
        dt: DataTree = DataTree(data=ds)
        assert ds.mean().identical(dt.ds.mean())
        assert type(dt.ds.mean()) == xr.Dataset

    def test_arithmetic(self, create_test_datatree):
        dt = create_test_datatree()
        expected = create_test_datatree(modify=lambda ds: 10.0 * ds)['set1']
        result = 10.0 * dt['set1'].ds
        assert result.identical(expected)

    def test_init_via_type(self):
        a = xr.DataArray(np.random.rand(3, 4, 10), dims=['x', 'y', 'time'], coords={'area': (['x', 'y'], np.random.rand(3, 4))}).to_dataset(name='data')
        dt: DataTree = DataTree(data=a)

        def weighted_mean(ds):
            return ds.weighted(ds.area).mean(['x', 'y'])
        weighted_mean(dt.ds)
from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestRestructuring:

    def test_drop_nodes(self):
        sue = DataTree.from_dict({'Mary': None, 'Kate': None, 'Ashley': None})
        dropped_one = sue.drop_nodes(names='Mary')
        assert 'Mary' not in dropped_one.children
        dropped = sue.drop_nodes(names=['Mary', 'Kate'])
        assert not set(['Mary', 'Kate']).intersection(set(dropped.children))
        assert 'Ashley' in dropped.children
        with pytest.raises(KeyError, match="nodes {'Mary'} not present"):
            dropped.drop_nodes(names=['Mary', 'Ashley'])
        childless = dropped.drop_nodes(names=['Mary', 'Ashley'], errors='ignore')
        assert childless.children == {}

    def test_assign(self):
        dt: DataTree = DataTree()
        expected = DataTree.from_dict({'/': xr.Dataset({'foo': 0}), '/a': None})
        result = dt.assign(foo=xr.DataArray(0), a=DataTree())
        dtt.assert_equal(result, expected)
        result = dt.assign({'foo': xr.DataArray(0), 'a': DataTree()})
        dtt.assert_equal(result, expected)
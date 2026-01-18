from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
class TestSubset:

    def test_match(self):
        dt = DataTree.from_dict({'/a/A': None, '/a/B': None, '/b/A': None, '/b/B': None})
        result = dt.match('*/B')
        expected = DataTree.from_dict({'/a/B': None, '/b/B': None})
        dtt.assert_identical(result, expected)

    def test_filter(self):
        simpsons = DataTree.from_dict(d={'/': xr.Dataset({'age': 83}), '/Herbert': xr.Dataset({'age': 40}), '/Homer': xr.Dataset({'age': 39}), '/Homer/Bart': xr.Dataset({'age': 10}), '/Homer/Lisa': xr.Dataset({'age': 8}), '/Homer/Maggie': xr.Dataset({'age': 1})}, name='Abe')
        expected = DataTree.from_dict(d={'/': xr.Dataset({'age': 83}), '/Herbert': xr.Dataset({'age': 40}), '/Homer': xr.Dataset({'age': 39})}, name='Abe')
        elders = simpsons.filter(lambda node: node['age'].item() > 18)
        dtt.assert_identical(elders, expected)
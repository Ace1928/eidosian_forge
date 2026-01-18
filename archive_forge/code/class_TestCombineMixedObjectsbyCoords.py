from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
class TestCombineMixedObjectsbyCoords:

    def test_combine_by_coords_mixed_unnamed_dataarrays(self):
        named_da = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        unnamed_da = DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
        with pytest.raises(ValueError, match="Can't automatically combine unnamed DataArrays with"):
            combine_by_coords([named_da, unnamed_da])
        da = DataArray([0, 1], dims='x', coords={'x': [0, 1]})
        ds = Dataset({'x': [2, 3]})
        with pytest.raises(ValueError, match="Can't automatically combine unnamed DataArrays with"):
            combine_by_coords([da, ds])

    def test_combine_coords_mixed_datasets_named_dataarrays(self):
        da = DataArray(name='a', data=[4, 5], dims='x', coords={'x': [0, 1]})
        ds = Dataset({'b': ('x', [2, 3])})
        actual = combine_by_coords([da, ds])
        expected = Dataset({'a': ('x', [4, 5]), 'b': ('x', [2, 3])}, coords={'x': ('x', [0, 1])})
        assert_identical(expected, actual)

    def test_combine_by_coords_all_unnamed_dataarrays(self):
        unnamed_array = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        actual = combine_by_coords([unnamed_array])
        expected = unnamed_array
        assert_identical(expected, actual)
        unnamed_array1 = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        unnamed_array2 = DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
        actual = combine_by_coords([unnamed_array1, unnamed_array2])
        expected = DataArray(data=[1.0, 2.0, 3.0, 4.0], coords={'x': [0, 1, 2, 3]}, dims='x')
        assert_identical(expected, actual)

    def test_combine_by_coords_all_named_dataarrays(self):
        named_da = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        actual = combine_by_coords([named_da])
        expected = named_da.to_dataset()
        assert_identical(expected, actual)
        named_da1 = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        named_da2 = DataArray(name='b', data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
        actual = combine_by_coords([named_da1, named_da2])
        expected = Dataset({'a': DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x'), 'b': DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')})
        assert_identical(expected, actual)

    def test_combine_by_coords_all_dataarrays_with_the_same_name(self):
        named_da1 = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
        named_da2 = DataArray(name='a', data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
        actual = combine_by_coords([named_da1, named_da2])
        expected = merge([named_da1, named_da2])
        assert_identical(expected, actual)
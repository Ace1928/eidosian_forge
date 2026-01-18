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
class TestCombineDatasetsbyCoords:

    def test_combine_by_coords(self):
        objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': [0, 1]})
        assert_identical(expected, actual)
        actual = combine_by_coords([actual])
        assert_identical(expected, actual)
        objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': [0, 1, 2]})
        assert_identical(expected, actual)
        objs = [Dataset({'x': ('a', [0]), 'y': ('a', [0]), 'a': [0]}), Dataset({'x': ('a', [1]), 'y': ('a', [1]), 'a': [1]})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1]), 'a': [0, 1]})
        assert_identical(expected, actual)
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': [0, 1], 'y': [0, 1]})
        assert_equal(actual, expected)
        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        with pytest.raises(ValueError, match='Could not find any dimension coordinates'):
            combine_by_coords(objs)
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
        with pytest.raises(ValueError, match='Every dimension needs a coordinate'):
            combine_by_coords(objs)

    def test_empty_input(self):
        assert_identical(Dataset(), combine_by_coords([]))

    @pytest.mark.parametrize('join, expected', [('outer', Dataset({'x': [0, 1], 'y': [0, 1]})), ('inner', Dataset({'x': [0, 1], 'y': []})), ('left', Dataset({'x': [0, 1], 'y': [0]})), ('right', Dataset({'x': [0, 1], 'y': [1]}))])
    def test_combine_coords_join(self, join, expected):
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [1], 'y': [1]})]
        actual = combine_nested(objs, concat_dim='x', join=join)
        assert_identical(expected, actual)

    def test_combine_coords_join_exact(self):
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [1], 'y': [1]})]
        with pytest.raises(ValueError, match='cannot align.*join.*exact.*'):
            combine_nested(objs, concat_dim='x', join='exact')

    @pytest.mark.parametrize('combine_attrs, expected', [('drop', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={})), ('no_conflicts', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'b': 2})), ('override', Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1})), (lambda attrs, context: attrs[1], Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'b': 2}))])
    def test_combine_coords_combine_attrs(self, combine_attrs, expected):
        objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1, 'b': 2})]
        actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs=combine_attrs)
        assert_identical(expected, actual)
        if combine_attrs == 'no_conflicts':
            objs[1].attrs['a'] = 2
            with pytest.raises(ValueError, match="combine_attrs='no_conflicts'"):
                actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs=combine_attrs)

    def test_combine_coords_combine_attrs_identical(self):
        objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1})]
        expected = Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1})
        actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='identical')
        assert_identical(expected, actual)
        objs[1].attrs['b'] = 2
        with pytest.raises(ValueError, match="combine_attrs='identical'"):
            actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='identical')

    def test_combine_nested_combine_attrs_drop_conflicts(self):
        objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1, 'b': 2, 'c': 3}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1, 'b': 0, 'd': 3})]
        expected = Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'c': 3, 'd': 3})
        actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='drop_conflicts')
        assert_identical(expected, actual)

    @pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False)])
    def test_combine_nested_combine_attrs_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
        """check that combine_attrs is used on data variables and coords"""
        data1 = Dataset({'a': ('x', [1, 2], attrs1), 'b': ('x', [3, -1], attrs1), 'x': ('x', [0, 1], attrs1)})
        data2 = Dataset({'a': ('x', [2, 3], attrs2), 'b': ('x', [-2, 1], attrs2), 'x': ('x', [2, 3], attrs2)})
        if expect_exception:
            with pytest.raises(MergeError, match='combine_attrs'):
                combine_by_coords([data1, data2], combine_attrs=combine_attrs)
        else:
            actual = combine_by_coords([data1, data2], combine_attrs=combine_attrs)
            expected = Dataset({'a': ('x', [1, 2, 2, 3], expected_attrs), 'b': ('x', [3, -1, -2, 1], expected_attrs)}, {'x': ('x', [0, 1, 2, 3], expected_attrs)})
            assert_identical(actual, expected)

    @pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False)])
    def test_combine_by_coords_combine_attrs_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
        """check that combine_attrs is used on data variables and coords"""
        data1 = Dataset({'x': ('a', [0], attrs1), 'y': ('a', [0], attrs1), 'a': ('a', [0], attrs1)})
        data2 = Dataset({'x': ('a', [1], attrs2), 'y': ('a', [1], attrs2), 'a': ('a', [1], attrs2)})
        if expect_exception:
            with pytest.raises(MergeError, match='combine_attrs'):
                combine_by_coords([data1, data2], combine_attrs=combine_attrs)
        else:
            actual = combine_by_coords([data1, data2], combine_attrs=combine_attrs)
            expected = Dataset({'x': ('a', [0, 1], expected_attrs), 'y': ('a', [0, 1], expected_attrs), 'a': ('a', [0, 1], expected_attrs)})
            assert_identical(actual, expected)

    def test_infer_order_from_coords(self):
        data = create_test_data()
        objs = [data.isel(dim2=slice(4, 9)), data.isel(dim2=slice(4))]
        actual = combine_by_coords(objs)
        expected = data
        assert expected.broadcast_equals(actual)

    def test_combine_leaving_bystander_dimensions(self):
        ycoord = ['a', 'c', 'b']
        data = np.random.rand(7, 3)
        ds1 = Dataset(data_vars=dict(data=(['x', 'y'], data[:3, :])), coords=dict(x=[1, 2, 3], y=ycoord))
        ds2 = Dataset(data_vars=dict(data=(['x', 'y'], data[3:, :])), coords=dict(x=[4, 5, 6, 7], y=ycoord))
        expected = Dataset(data_vars=dict(data=(['x', 'y'], data)), coords=dict(x=[1, 2, 3, 4, 5, 6, 7], y=ycoord))
        actual = combine_by_coords((ds1, ds2))
        assert_identical(expected, actual)

    def test_combine_by_coords_previously_failed(self):
        datasets = [Dataset({'a': ('x', [0]), 'x': [0]}), Dataset({'b': ('x', [0]), 'x': [0]}), Dataset({'a': ('x', [1]), 'x': [1]})]
        expected = Dataset({'a': ('x', [0, 1]), 'b': ('x', [0, np.nan])}, {'x': [0, 1]})
        actual = combine_by_coords(datasets)
        assert_identical(expected, actual)

    def test_combine_by_coords_still_fails(self):
        datasets = [Dataset({'x': 0}, {'y': 0}), Dataset({'x': 1}, {'y': 1, 'z': 1})]
        with pytest.raises(ValueError):
            combine_by_coords(datasets, 'y')

    def test_combine_by_coords_no_concat(self):
        objs = [Dataset({'x': 0}), Dataset({'y': 1})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': 0, 'y': 1})
        assert_identical(expected, actual)
        objs = [Dataset({'x': 0, 'y': 1}), Dataset({'y': np.nan, 'z': 2})]
        actual = combine_by_coords(objs)
        expected = Dataset({'x': 0, 'y': 1, 'z': 2})
        assert_identical(expected, actual)

    def test_check_for_impossible_ordering(self):
        ds0 = Dataset({'x': [0, 1, 5]})
        ds1 = Dataset({'x': [2, 3]})
        with pytest.raises(ValueError, match='does not have monotonic global indexes along dimension x'):
            combine_by_coords([ds1, ds0])

    def test_combine_by_coords_incomplete_hypercube(self):
        x1 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [0], 'x': [0]})
        x2 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [1], 'x': [0]})
        x3 = Dataset({'a': (('y', 'x'), [[1]])}, coords={'y': [0], 'x': [1]})
        actual = combine_by_coords([x1, x2, x3])
        expected = Dataset({'a': (('y', 'x'), [[1, 1], [1, np.nan]])}, coords={'y': [0, 1], 'x': [0, 1]})
        assert_identical(expected, actual)
        with pytest.raises(ValueError):
            combine_by_coords([x1, x2, x3], fill_value=None)
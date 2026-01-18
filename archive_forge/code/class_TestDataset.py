from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
class TestDataset:

    @pytest.mark.parametrize('unit,error', (pytest.param(1, xr.MergeError, id='no_unit'), pytest.param(unit_registry.dimensionless, xr.MergeError, id='dimensionless'), pytest.param(unit_registry.s, xr.MergeError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='same_unit')))
    @pytest.mark.parametrize('shared', ('nothing', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_init(self, shared, unit, error, dtype):
        original_unit = unit_registry.m
        scaled_unit = unit_registry.mm
        a = np.linspace(0, 1, 10).astype(dtype) * unit_registry.Pa
        b = np.linspace(-1, 0, 10).astype(dtype) * unit_registry.degK
        values_a = np.arange(a.shape[0])
        dim_a = values_a * original_unit
        coord_a = dim_a.to(scaled_unit)
        values_b = np.arange(b.shape[0])
        dim_b = values_b * unit
        coord_b = dim_b.to(scaled_unit) if unit_registry.is_compatible_with(dim_b, scaled_unit) and unit != scaled_unit else dim_b * 1000
        variants = {'nothing': ({}, {}), 'dims': ({'x': dim_a}, {'x': dim_b}), 'coords': ({'x': values_a, 'y': ('x', coord_a)}, {'x': values_b, 'y': ('x', coord_b)})}
        coords_a, coords_b = variants.get(shared)
        dims_a, dims_b = ('x', 'y') if shared == 'nothing' else ('x', 'x')
        a = xr.DataArray(data=a, coords=coords_a, dims=dims_a)
        b = xr.DataArray(data=b, coords=coords_b, dims=dims_b)
        if error is not None and shared != 'nothing':
            with pytest.raises(error):
                xr.Dataset(data_vars={'a': a, 'b': b})
            return
        actual = xr.Dataset(data_vars={'a': a, 'b': b})
        units = merge_mappings(extract_units(a.rename('a')), extract_units(b.rename('b')))
        expected = attach_units(xr.Dataset(data_vars={'a': strip_units(a), 'b': strip_units(b)}), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (pytest.param(str, id='str'), pytest.param(repr, id='repr')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_repr(self, func, variant, dtype):
        unit1, unit2 = (unit_registry.Pa, unit_registry.degK) if variant == 'data' else (1, 1)
        array1 = np.linspace(1, 2, 10, dtype=dtype) * unit1
        array2 = np.linspace(0, 1, 10, dtype=dtype) * unit2
        x = np.arange(len(array1)) * unit_registry.s
        y = x.to(unit_registry.ms)
        variants = {'dims': {'x': x}, 'coords': {'y': ('x', y)}, 'data': {}}
        ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('x', array2)}, coords=variants.get(variant))
        func(ds)

    @pytest.mark.parametrize('func', (method('all'), method('any'), method('argmax', dim='x'), method('argmin', dim='x'), method('max'), method('min'), method('mean'), method('median'), method('sum'), method('prod'), method('std'), method('var'), method('cumsum'), method('cumprod')), ids=repr)
    def test_aggregation(self, func, dtype):
        unit_a, unit_b = (unit_registry.Pa, unit_registry.degK) if func.name != 'cumprod' else (unit_registry.dimensionless, unit_registry.dimensionless)
        a = np.linspace(0, 1, 10).astype(dtype) * unit_a
        b = np.linspace(-1, 0, 10).astype(dtype) * unit_b
        ds = xr.Dataset({'a': ('x', a), 'b': ('x', b)})
        if 'dim' in func.kwargs:
            numpy_kwargs = func.kwargs.copy()
            dim = numpy_kwargs.pop('dim')
            axis_a = ds.a.get_axis_num(dim)
            axis_b = ds.b.get_axis_num(dim)
            numpy_kwargs_a = numpy_kwargs.copy()
            numpy_kwargs_a['axis'] = axis_a
            numpy_kwargs_b = numpy_kwargs.copy()
            numpy_kwargs_b['axis'] = axis_b
        else:
            numpy_kwargs_a = {}
            numpy_kwargs_b = {}
        units_a = array_extract_units(func(a, **numpy_kwargs_a))
        units_b = array_extract_units(func(b, **numpy_kwargs_b))
        units = {'a': units_a, 'b': units_b}
        actual = func(ds)
        expected = attach_units(func(strip_units(ds)), units)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('property', ('imag', 'real'))
    def test_numpy_properties(self, property, dtype):
        a = np.linspace(0, 1, 10) * unit_registry.Pa
        b = np.linspace(-1, 0, 15) * unit_registry.degK
        ds = xr.Dataset({'a': ('x', a), 'b': ('y', b)})
        units = extract_units(ds)
        actual = getattr(ds, property)
        expected = attach_units(getattr(strip_units(ds), property), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('astype', float), method('conj'), method('argsort'), method('conjugate'), method('round')), ids=repr)
    def test_numpy_methods(self, func, dtype):
        a = np.linspace(1, -1, 10) * unit_registry.Pa
        b = np.linspace(-1, 1, 15) * unit_registry.degK
        ds = xr.Dataset({'a': ('x', a), 'b': ('y', b)})
        units_a = array_extract_units(func(a))
        units_b = array_extract_units(func(b))
        units = {'a': units_a, 'b': units_b}
        actual = func(ds)
        expected = attach_units(func(strip_units(ds)), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('clip', min=3, max=8),), ids=repr)
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_numpy_methods_with_args(self, func, unit, error, dtype):
        data_unit = unit_registry.m
        a = np.linspace(0, 10, 15) * unit_registry.m
        b = np.linspace(-2, 12, 20) * unit_registry.m
        ds = xr.Dataset({'a': ('x', a), 'b': ('y', b)})
        units = extract_units(ds)
        kwargs = {key: array_attach_units(value, unit) for key, value in func.kwargs.items()}
        if error is not None:
            with pytest.raises(error):
                func(ds, **kwargs)
            return
        stripped_kwargs = {key: strip_units(convert_units(value, {None: data_unit})) for key, value in kwargs.items()}
        actual = func(ds, **kwargs)
        expected = attach_units(func(strip_units(ds), **stripped_kwargs), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('isnull'), method('notnull'), method('count')), ids=repr)
    def test_missing_value_detection(self, func, dtype):
        array1 = np.array([[1.4, 2.3, np.nan, 7.2], [np.nan, 9.7, np.nan, np.nan], [2.1, np.nan, np.nan, 4.6], [9.9, np.nan, 7.2, 9.1]]) * unit_registry.degK
        array2 = np.array([[np.nan, 5.7, 12.0, 7.2], [np.nan, 12.4, np.nan, 4.2], [9.8, np.nan, 4.6, 1.4], [7.2, np.nan, 6.3, np.nan], [8.4, 3.9, np.nan, np.nan]]) * unit_registry.Pa
        ds = xr.Dataset({'a': (('x', 'y'), array1), 'b': (('z', 'x'), array2)})
        expected = func(strip_units(ds))
        actual = func(ds)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.xfail(reason='ffill and bfill lose the unit')
    @pytest.mark.parametrize('func', (method('ffill'), method('bfill')), ids=repr)
    def test_missing_value_filling(self, func, dtype):
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.degK
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.Pa
        ds = xr.Dataset({'a': ('x', array1), 'b': ('y', array2)})
        units = extract_units(ds)
        expected = attach_units(func(strip_units(ds), dim='x'), units)
        actual = func(ds, dim='x')
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('fill_value', (pytest.param(-1, id='python_scalar'), pytest.param(np.array(-1), id='numpy_scalar'), pytest.param(np.array([-1]), id='numpy_array')))
    def test_fillna(self, fill_value, unit, error, dtype):
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.m
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.m
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
        value = fill_value * unit
        units = extract_units(ds)
        if error is not None:
            with pytest.raises(error):
                ds.fillna(value=value)
            return
        actual = ds.fillna(value=value)
        expected = attach_units(strip_units(ds).fillna(value=strip_units(convert_units(value, {None: unit_registry.m}))), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    def test_dropna(self, dtype):
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.degK
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.Pa
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
        units = extract_units(ds)
        expected = attach_units(strip_units(ds).dropna(dim='x'), units)
        actual = ds.dropna(dim='x')
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='same_unit')))
    def test_isin(self, unit, dtype):
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.m
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.m
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
        raw_values = np.array([1.4, np.nan, 2.3]).astype(dtype)
        values = raw_values * unit
        converted_values = convert_units(values, {None: unit_registry.m}) if is_compatible(unit, unit_registry.m) else values
        expected = strip_units(ds).isin(strip_units(converted_values))
        if not is_compatible(unit, unit_registry.m):
            expected.a[:] = False
            expected.b[:] = False
        actual = ds.isin(values)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('variant', ('masking', 'replacing_scalar', 'replacing_array', 'dropping'))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='same_unit')))
    def test_where(self, variant, unit, error, dtype):
        original_unit = unit_registry.m
        array1 = np.linspace(0, 1, 10).astype(dtype) * original_unit
        array2 = np.linspace(-1, 0, 10).astype(dtype) * original_unit
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
        units = extract_units(ds)
        condition = ds < 0.5 * original_unit
        other = np.linspace(-2, -1, 10).astype(dtype) * unit
        variant_kwargs = {'masking': {'cond': condition}, 'replacing_scalar': {'cond': condition, 'other': -1 * unit}, 'replacing_array': {'cond': condition, 'other': other}, 'dropping': {'cond': condition, 'drop': True}}
        kwargs = variant_kwargs.get(variant)
        if variant not in ('masking', 'dropping') and error is not None:
            with pytest.raises(error):
                ds.where(**kwargs)
            return
        kwargs_without_units = {key: strip_units(convert_units(value, {None: original_unit})) for key, value in kwargs.items()}
        expected = attach_units(strip_units(ds).where(**kwargs_without_units), units)
        actual = ds.where(**kwargs)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.xfail(reason='interpolate_na uses numpy.vectorize')
    def test_interpolate_na(self, dtype):
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.degK
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.Pa
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
        units = extract_units(ds)
        expected = attach_units(strip_units(ds).interpolate_na(dim='x'), units)
        actual = ds.interpolate_na(dim='x')
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='same_unit')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
    def test_combine_first(self, variant, unit, error, dtype):
        variants = {'data': (unit_registry.m, unit, 1, 1), 'dims': (1, 1, unit_registry.m, unit)}
        data_unit, other_data_unit, dims_unit, other_dims_unit = variants.get(variant)
        array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * data_unit
        array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * data_unit
        x = np.arange(len(array1)) * dims_unit
        ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x})
        units = extract_units(ds)
        other_array1 = np.ones_like(array1) * other_data_unit
        other_array2 = np.full_like(array2, fill_value=-1) * other_data_unit
        other_x = (np.arange(array1.shape[0]) + 5) * other_dims_unit
        other = xr.Dataset(data_vars={'a': ('x', other_array1), 'b': ('x', other_array2)}, coords={'x': other_x})
        if error is not None:
            with pytest.raises(error):
                ds.combine_first(other)
            return
        expected = attach_units(strip_units(ds).combine_first(strip_units(convert_units(other, units))), units)
        actual = ds.combine_first(other)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('equals'), pytest.param(method('identical'), marks=pytest.mark.skip('behaviour of identical is unclear'))), ids=repr)
    def test_comparisons(self, func, variant, unit, dtype):
        array1 = np.linspace(0, 5, 10).astype(dtype)
        array2 = np.linspace(-5, 0, 10).astype(dtype)
        coord = np.arange(len(array1)).astype(dtype)
        variants = {'data': (unit_registry.m, 1, 1), 'dims': (1, unit_registry.m, 1), 'coords': (1, 1, unit_registry.m)}
        data_unit, dim_unit, coord_unit = variants.get(variant)
        a = array1 * data_unit
        b = array2 * data_unit
        x = coord * dim_unit
        y = coord * coord_unit
        ds = xr.Dataset(data_vars={'a': ('x', a), 'b': ('x', b)}, coords={'x': x, 'y': ('x', y)})
        units = extract_units(ds)
        other_variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        other_data_unit, other_dim_unit, other_coord_unit = other_variants.get(variant)
        other_units = {'a': other_data_unit, 'b': other_data_unit, 'x': other_dim_unit, 'y': other_coord_unit}
        to_convert = {key: unit if is_compatible(unit, reference) else None for key, (unit, reference) in zip_mappings(units, other_units)}
        other = attach_units(strip_units(convert_units(ds, to_convert)), other_units)
        other_units = extract_units(other)
        equal_ds = all((is_compatible(unit, other_unit) for _, (unit, other_unit) in zip_mappings(units, other_units))) and strip_units(ds).equals(strip_units(convert_units(other, units)))
        equal_units = units == other_units
        expected = equal_ds and (func.name != 'identical' or equal_units)
        actual = func(ds, other)
        assert expected == actual

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
    def test_broadcast_like(self, variant, unit, dtype):
        variants = {'data': ((unit_registry.m, unit), (1, 1)), 'dims': ((1, 1), (unit_registry.m, unit))}
        (data_unit1, data_unit2), (dim_unit1, dim_unit2) = variants.get(variant)
        array1 = np.linspace(1, 2, 2 * 1).reshape(2, 1).astype(dtype) * data_unit1
        array2 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * data_unit2
        x1 = np.arange(2) * dim_unit1
        x2 = np.arange(2) * dim_unit2
        y1 = np.array([0]) * dim_unit1
        y2 = np.arange(3) * dim_unit2
        ds1 = xr.Dataset(data_vars={'a': (('x', 'y'), array1)}, coords={'x': x1, 'y': y1})
        ds2 = xr.Dataset(data_vars={'a': (('x', 'y'), array2)}, coords={'x': x2, 'y': y2})
        expected = attach_units(strip_units(ds1).broadcast_like(strip_units(ds2)), extract_units(ds1))
        actual = ds1.broadcast_like(ds2)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    def test_broadcast_equals(self, unit, dtype):
        left_array1 = np.ones(shape=(2, 3), dtype=dtype) * unit_registry.m
        left_array2 = np.zeros(shape=(3, 6), dtype=dtype) * unit_registry.m
        right_array1 = np.ones(shape=(2,)) * unit
        right_array2 = np.zeros(shape=(3,)) * unit
        left = xr.Dataset({'a': (('x', 'y'), left_array1), 'b': (('y', 'z'), left_array2)})
        right = xr.Dataset({'a': ('x', right_array1), 'b': ('y', right_array2)})
        units = merge_mappings(extract_units(left), {} if is_compatible(left_array1, unit) else {'a': None, 'b': None})
        expected = is_compatible(left_array1, unit) and strip_units(left).broadcast_equals(strip_units(convert_units(right, units)))
        actual = left.broadcast_equals(right)
        assert expected == actual

    def test_pad(self, dtype):
        a = np.linspace(0, 5, 10).astype(dtype) * unit_registry.Pa
        b = np.linspace(-5, 0, 10).astype(dtype) * unit_registry.degK
        ds = xr.Dataset({'a': ('x', a), 'b': ('x', b)})
        units = extract_units(ds)
        expected = attach_units(strip_units(ds).pad(x=(2, 3)), units)
        actual = ds.pad(x=(2, 3))
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('unstack'), method('reset_index', 'v'), method('reorder_levels')), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
    def test_stacking_stacked(self, variant, func, dtype):
        variants = {'data': (unit_registry.m, 1), 'dims': (1, unit_registry.m)}
        data_unit, dim_unit = variants.get(variant)
        array1 = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        array2 = np.linspace(-10, 0, 5 * 10 * 15).reshape(5, 10, 15).astype(dtype) * data_unit
        x = np.arange(array1.shape[0]) * dim_unit
        y = np.arange(array1.shape[1]) * dim_unit
        z = np.arange(array2.shape[2]) * dim_unit
        ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'y', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z})
        units = extract_units(ds)
        stacked = ds.stack(v=('x', 'y'))
        expected = attach_units(func(strip_units(stacked)), units)
        actual = func(stacked)
        assert_units_equal(expected, actual)
        if func.name == 'reset_index':
            assert_equal(expected, actual, check_default_indexes=False)
        else:
            assert_equal(expected, actual)

    @pytest.mark.xfail(reason="stacked dimension's labels have to be hashable, but is a numpy.array")
    def test_to_stacked_array(self, dtype):
        labels = range(5) * unit_registry.s
        arrays = {name: np.linspace(0, 1, 10).astype(dtype) * unit_registry.m for name in labels}
        ds = xr.Dataset({name: ('x', array) for name, array in arrays.items()})
        units = {None: unit_registry.m, 'y': unit_registry.s}
        func = method('to_stacked_array', 'z', variable_dim='y', sample_dims=['x'])
        actual = func(ds).rename(None)
        expected = attach_units(func(strip_units(ds)).rename(None), units)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('transpose', 'y', 'x', 'z1', 'z2'), method('stack', u=('x', 'y')), method('set_index', x='x2'), method('shift', x=2), pytest.param(method('rank', dim='x'), marks=pytest.mark.skip(reason='rank not implemented for non-ndarray')), method('roll', x=2, roll_coords=False), method('sortby', 'x2')), ids=repr)
    def test_stacking_reordering(self, func, dtype):
        array1 = np.linspace(0, 10, 2 * 5 * 10).reshape(2, 5, 10).astype(dtype) * unit_registry.Pa
        array2 = np.linspace(0, 10, 2 * 5 * 15).reshape(2, 5, 15).astype(dtype) * unit_registry.degK
        x = np.arange(array1.shape[0])
        y = np.arange(array1.shape[1])
        z1 = np.arange(array1.shape[2])
        z2 = np.arange(array2.shape[2])
        x2 = np.linspace(0, 1, array1.shape[0])[::-1]
        ds = xr.Dataset(data_vars={'a': (('x', 'y', 'z1'), array1), 'b': (('x', 'y', 'z2'), array2)}, coords={'x': x, 'y': y, 'z1': z1, 'z2': z2, 'x2': ('x', x2)})
        units = extract_units(ds)
        expected = attach_units(func(strip_units(ds)), units)
        actual = func(ds)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('indices', (pytest.param(4, id='single index'), pytest.param([5, 2, 9, 1], id='multiple indices')))
    def test_isel(self, indices, dtype):
        array1 = np.arange(10).astype(dtype) * unit_registry.s
        array2 = np.linspace(0, 1, 10).astype(dtype) * unit_registry.Pa
        ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('x', array2)})
        units = extract_units(ds)
        expected = attach_units(strip_units(ds).isel(x=indices), units)
        actual = ds.isel(x=indices)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.mm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_sel(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m
        ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims='x'), 'b': xr.DataArray(data=array2, dims='x')}, coords={'x': x})
        values = raw_values * unit
        if error is not None:
            with pytest.raises(error):
                ds.sel(x=values)
            return
        expected = attach_units(strip_units(ds).sel(x=strip_units(convert_units(values, {None: unit_registry.m}))), extract_units(ds))
        actual = ds.sel(x=values)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.mm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_drop_sel(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m
        ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims='x'), 'b': xr.DataArray(data=array2, dims='x')}, coords={'x': x})
        values = raw_values * unit
        if error is not None:
            with pytest.raises(error):
                ds.drop_sel(x=values)
            return
        expected = attach_units(strip_units(ds).drop_sel(x=strip_units(convert_units(values, {None: unit_registry.m}))), extract_units(ds))
        actual = ds.drop_sel(x=values)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.mm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_loc(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m
        ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims='x'), 'b': xr.DataArray(data=array2, dims='x')}, coords={'x': x})
        values = raw_values * unit
        if error is not None:
            with pytest.raises(error):
                ds.loc[{'x': values}]
            return
        expected = attach_units(strip_units(ds).loc[{'x': strip_units(convert_units(values, {None: unit_registry.m}))}], extract_units(ds))
        actual = ds.loc[{'x': values}]
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('head', x=7, y=3, z=6), method('tail', x=7, y=3, z=6), method('thin', x=7, y=3, z=6)), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_head_tail_thin(self, func, variant, dtype):
        variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
        (unit_a, unit_b), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_a
        array2 = np.linspace(1, 2, 10 * 8).reshape(10, 8) * unit_b
        coords = {'x': np.arange(10) * dim_unit, 'y': np.arange(5) * dim_unit, 'z': np.arange(8) * dim_unit, 'u': ('x', np.linspace(0, 1, 10) * coord_unit), 'v': ('y', np.linspace(1, 2, 5) * coord_unit), 'w': ('z', np.linspace(-1, 0, 8) * coord_unit)}
        ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims=('x', 'y')), 'b': xr.DataArray(data=array2, dims=('x', 'z'))}, coords=coords)
        expected = attach_units(func(strip_units(ds)), extract_units(ds))
        actual = func(ds)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('dim', ('x', 'y', 'z', 't', 'all'))
    @pytest.mark.parametrize('shape', (pytest.param((10, 20), id='nothing squeezable'), pytest.param((10, 20, 1), id='last dimension squeezable'), pytest.param((10, 1, 20), id='middle dimension squeezable'), pytest.param((1, 10, 20), id='first dimension squeezable'), pytest.param((1, 10, 1, 20), id='first and last dimension squeezable')))
    def test_squeeze(self, shape, dim, dtype):
        names = 'xyzt'
        dim_lengths = dict(zip(names, shape))
        array1 = np.linspace(0, 1, 10 * 20).astype(dtype).reshape(shape) * unit_registry.degK
        array2 = np.linspace(1, 2, 10 * 20).astype(dtype).reshape(shape) * unit_registry.Pa
        ds = xr.Dataset(data_vars={'a': (tuple(names[:len(shape)]), array1), 'b': (tuple(names[:len(shape)]), array2)})
        units = extract_units(ds)
        kwargs = {'dim': dim} if dim != 'all' and dim_lengths.get(dim, 0) == 1 else {}
        expected = attach_units(strip_units(ds).squeeze(**kwargs), units)
        actual = ds.squeeze(**kwargs)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('variant', ('data', 'coords'))
    @pytest.mark.parametrize('func', (pytest.param(method('interp'), marks=pytest.mark.xfail(reason='uses scipy')), method('reindex')), ids=repr)
    def test_interp_reindex(self, func, variant, dtype):
        variants = {'data': (unit_registry.m, 1), 'coords': (1, unit_registry.m)}
        data_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-1, 0, 10).astype(dtype) * data_unit
        array2 = np.linspace(0, 1, 10).astype(dtype) * data_unit
        y = np.arange(10) * coord_unit
        x = np.arange(10)
        new_x = np.arange(8) + 0.5
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x, 'y': ('x', y)})
        units = extract_units(ds)
        expected = attach_units(func(strip_units(ds), x=new_x), units)
        actual = func(ds, x=new_x)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('interp'), method('reindex')), ids=repr)
    def test_interp_reindex_indexing(self, func, unit, error, dtype):
        array1 = np.linspace(-1, 0, 10).astype(dtype)
        array2 = np.linspace(0, 1, 10).astype(dtype)
        x = np.arange(10) * unit_registry.m
        new_x = (np.arange(8) + 0.5) * unit
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x})
        units = extract_units(ds)
        if error is not None:
            with pytest.raises(error):
                func(ds, x=new_x)
            return
        expected = attach_units(func(strip_units(ds), x=new_x), units)
        actual = func(ds, x=new_x)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('variant', ('data', 'coords'))
    @pytest.mark.parametrize('func', (pytest.param(method('interp_like'), marks=pytest.mark.xfail(reason='uses scipy')), method('reindex_like')), ids=repr)
    def test_interp_reindex_like(self, func, variant, dtype):
        variants = {'data': (unit_registry.m, 1), 'coords': (1, unit_registry.m)}
        data_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-1, 0, 10).astype(dtype) * data_unit
        array2 = np.linspace(0, 1, 10).astype(dtype) * data_unit
        y = np.arange(10) * coord_unit
        x = np.arange(10)
        new_x = np.arange(8) + 0.5
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x, 'y': ('x', y)})
        units = extract_units(ds)
        other = xr.Dataset({'a': ('x', np.empty_like(new_x))}, coords={'x': new_x})
        expected = attach_units(func(strip_units(ds), other), units)
        actual = func(ds, other)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('interp_like'), method('reindex_like')), ids=repr)
    def test_interp_reindex_like_indexing(self, func, unit, error, dtype):
        array1 = np.linspace(-1, 0, 10).astype(dtype)
        array2 = np.linspace(0, 1, 10).astype(dtype)
        x = np.arange(10) * unit_registry.m
        new_x = (np.arange(8) + 0.5) * unit
        ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x})
        units = extract_units(ds)
        other = xr.Dataset({'a': ('x', np.empty_like(new_x))}, coords={'x': new_x})
        if error is not None:
            with pytest.raises(error):
                func(ds, other)
            return
        expected = attach_units(func(strip_units(ds), other), units)
        actual = func(ds, other)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
    @pytest.mark.parametrize('func', (method('diff', dim='x'), method('differentiate', coord='x'), method('integrate', coord='x'), method('quantile', q=[0.25, 0.75]), method('reduce', func=np.sum, dim='x'), method('map', np.fabs)), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_computation(self, func, variant, dtype, compute_backend):
        variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
        (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-5, 5, 4 * 5).reshape(4, 5).astype(dtype) * unit1
        array2 = np.linspace(10, 20, 4 * 3).reshape(4, 3).astype(dtype) * unit2
        x = np.arange(4) * dim_unit
        y = np.arange(5) * dim_unit
        z = np.arange(3) * dim_unit
        ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims=('x', 'y')), 'b': xr.DataArray(data=array2, dims=('x', 'z'))}, coords={'x': x, 'y': y, 'z': z, 'y2': ('y', np.arange(5) * coord_unit)})
        units = extract_units(ds)
        expected = attach_units(func(strip_units(ds)), units)
        actual = func(ds)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('groupby', 'x'), method('groupby_bins', 'x', bins=2), method('coarsen', x=2), pytest.param(method('rolling', x=3), marks=pytest.mark.xfail(reason='strips units')), pytest.param(method('rolling_exp', x=3), marks=pytest.mark.xfail(reason='numbagg functions are not supported by pint')), method('weighted', xr.DataArray(data=np.linspace(0, 1, 5), dims='y'))), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_computation_objects(self, func, variant, dtype):
        variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
        (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-5, 5, 4 * 5).reshape(4, 5).astype(dtype) * unit1
        array2 = np.linspace(10, 20, 4 * 3).reshape(4, 3).astype(dtype) * unit2
        x = np.arange(4) * dim_unit
        y = np.arange(5) * dim_unit
        z = np.arange(3) * dim_unit
        ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z, 'y2': ('y', np.arange(5) * coord_unit)})
        units = extract_units(ds)
        args = [] if func.name != 'groupby' else ['y']
        kwargs = {}
        expected = attach_units(func(strip_units(ds)).mean(*args, **kwargs), units)
        actual = func(ds).mean(*args, **kwargs)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_resample(self, variant, dtype):
        variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
        (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype) * unit1
        array2 = np.linspace(10, 20, 10 * 8).reshape(10, 8).astype(dtype) * unit2
        t = xr.date_range('10-09-2010', periods=array1.shape[0], freq='YE')
        y = np.arange(5) * dim_unit
        z = np.arange(8) * dim_unit
        u = np.linspace(-1, 0, 5) * coord_unit
        ds = xr.Dataset(data_vars={'a': (('time', 'y'), array1), 'b': (('time', 'z'), array2)}, coords={'time': t, 'y': y, 'z': z, 'u': ('y', u)})
        units = extract_units(ds)
        func = method('resample', time='6ME')
        expected = attach_units(func(strip_units(ds)).mean(), units)
        actual = func(ds).mean()
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
    @pytest.mark.parametrize('func', (method('assign', c=lambda ds: 10 * ds.b), method('assign_coords', v=('x', np.arange(5) * unit_registry.s)), method('first'), method('last'), method('quantile', q=[0.25, 0.5, 0.75], dim='x')), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_grouped_operations(self, func, variant, dtype, compute_backend):
        variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
        (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-5, 5, 5 * 4).reshape(5, 4).astype(dtype) * unit1
        array2 = np.linspace(10, 20, 5 * 4 * 3).reshape(5, 4, 3).astype(dtype) * unit2
        x = np.arange(5) * dim_unit
        y = np.arange(4) * dim_unit
        z = np.arange(3) * dim_unit
        u = np.linspace(-1, 0, 4) * coord_unit
        ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'y', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z, 'u': ('y', u)})
        assigned_units = {'c': unit2, 'v': unit_registry.s}
        units = merge_mappings(extract_units(ds), assigned_units)
        stripped_kwargs = {name: strip_units(value) for name, value in func.kwargs.items()}
        expected = attach_units(func(strip_units(ds).groupby('y', squeeze=False), **stripped_kwargs), units)
        actual = func(ds.groupby('y', squeeze=False))
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('func', (method('pipe', lambda ds: ds * 10), method('assign', d=lambda ds: ds.b * 10), method('assign_coords', y2=('y', np.arange(4) * unit_registry.mm)), method('assign_attrs', attr1='value'), method('rename', x2='x_mm'), method('rename_vars', c='temperature'), method('rename_dims', x='offset_x'), method('swap_dims', {'x': 'u'}), pytest.param(method('expand_dims', v=np.linspace(10, 20, 12) * unit_registry.s, axis=1), marks=pytest.mark.skip(reason="indexes don't support units")), method('drop_vars', 'x'), method('drop_dims', 'z'), method('set_coords', names='c'), method('reset_coords', names='x2'), method('copy')), ids=repr)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_content_manipulation(self, func, variant, dtype):
        variants = {'data': ((unit_registry.m ** 3, unit_registry.Pa, unit_registry.degK), 1, 1), 'dims': ((1, 1, 1), unit_registry.m, 1), 'coords': ((1, 1, 1), 1, unit_registry.m)}
        (unit1, unit2, unit3), dim_unit, coord_unit = variants.get(variant)
        array1 = np.linspace(-5, 5, 5 * 4).reshape(5, 4).astype(dtype) * unit1
        array2 = np.linspace(10, 20, 5 * 4 * 3).reshape(5, 4, 3).astype(dtype) * unit2
        array3 = np.linspace(0, 10, 5).astype(dtype) * unit3
        x = np.arange(5) * dim_unit
        y = np.arange(4) * dim_unit
        z = np.arange(3) * dim_unit
        x2 = np.linspace(-1, 0, 5) * coord_unit
        ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'y', 'z'), array2), 'c': ('x', array3)}, coords={'x': x, 'y': y, 'z': z, 'x2': ('x', x2)})
        new_units = {'y2': unit_registry.mm, 'x_mm': coord_unit, 'offset_x': unit_registry.m, 'd': unit2, 'temperature': unit3}
        units = merge_mappings(extract_units(ds), new_units)
        stripped_kwargs = {key: strip_units(value) for key, value in func.kwargs.items()}
        expected = attach_units(func(strip_units(ds), **stripped_kwargs), units)
        actual = func(ds)
        assert_units_equal(expected, actual)
        if func.name == 'rename_dims':
            assert_equal(expected, actual, check_default_indexes=False)
        else:
            assert_equal(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, xr.MergeError, id='no_unit'), pytest.param(unit_registry.dimensionless, xr.MergeError, id='dimensionless'), pytest.param(unit_registry.s, xr.MergeError, id='incompatible_unit'), pytest.param(unit_registry.cm, xr.MergeError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_merge(self, variant, unit, error, dtype):
        left_variants = {'data': (unit_registry.m, 1, 1), 'dims': (1, unit_registry.m, 1), 'coords': (1, 1, unit_registry.m)}
        left_data_unit, left_dim_unit, left_coord_unit = left_variants.get(variant)
        right_variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        right_data_unit, right_dim_unit, right_coord_unit = right_variants.get(variant)
        left_array = np.arange(10).astype(dtype) * left_data_unit
        right_array = np.arange(-5, 5).astype(dtype) * right_data_unit
        left_dim = np.arange(10, 20) * left_dim_unit
        right_dim = np.arange(5, 15) * right_dim_unit
        left_coord = np.arange(-10, 0) * left_coord_unit
        right_coord = np.arange(-15, -5) * right_coord_unit
        left = xr.Dataset(data_vars={'a': ('x', left_array)}, coords={'x': left_dim, 'y': ('x', left_coord)})
        right = xr.Dataset(data_vars={'a': ('x', right_array)}, coords={'x': right_dim, 'y': ('x', right_coord)})
        units = extract_units(left)
        if error is not None:
            with pytest.raises(error):
                left.merge(right)
            return
        converted = convert_units(right, units)
        expected = attach_units(strip_units(left).merge(strip_units(converted)), units)
        actual = left.merge(right)
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)
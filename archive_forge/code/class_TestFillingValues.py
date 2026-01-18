import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestFillingValues:

    def test_check_on_scalar(self):
        _check_fill_value = np.ma.core._check_fill_value
        fval = _check_fill_value(0, int)
        assert_equal(fval, 0)
        fval = _check_fill_value(None, int)
        assert_equal(fval, default_fill_value(0))
        fval = _check_fill_value(0, '|S3')
        assert_equal(fval, b'0')
        fval = _check_fill_value(None, '|S3')
        assert_equal(fval, default_fill_value(b'camelot!'))
        assert_raises(TypeError, _check_fill_value, 1e+20, int)
        assert_raises(TypeError, _check_fill_value, 'stuff', int)

    def test_check_on_fields(self):
        _check_fill_value = np.ma.core._check_fill_value
        ndtype = [('a', int), ('b', float), ('c', '|S3')]
        fval = _check_fill_value([-999, -12345678.9, '???'], ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b'???'])
        fval = _check_fill_value(None, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [default_fill_value(0), default_fill_value(0.0), asbytes(default_fill_value('0'))])
        fill_val = np.array((-999, -12345678.9, '???'), dtype=ndtype)
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b'???'])
        fill_val = np.array((-999, -12345678.9, '???'), dtype=[('A', int), ('B', float), ('C', '|S3')])
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b'???'])
        fill_val = np.ndarray(shape=(1,), dtype=object)
        fill_val[0] = (-999, -12345678.9, b'???')
        fval = _check_fill_value(fill_val, object)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b'???'])
        ndtype = [('a', int)]
        fval = _check_fill_value(-999999999, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), (-999999999,))

    def test_fillvalue_conversion(self):
        a = array([b'3', b'4', b'5'])
        a._optinfo.update({'comment': 'updated!'})
        b = array(a, dtype=int)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0))
        b = array(a, dtype=float)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0.0))
        b = a.astype(int)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0))
        assert_equal(b._optinfo['comment'], 'updated!')
        b = a.astype([('a', '|S3')])
        assert_equal(b['a']._data, a._data)
        assert_equal(b['a'].fill_value, a.fill_value)

    def test_default_fill_value(self):
        f1 = default_fill_value(1.0)
        f2 = default_fill_value(np.array(1.0))
        f3 = default_fill_value(np.array(1.0).dtype)
        assert_equal(f1, f2)
        assert_equal(f1, f3)

    def test_default_fill_value_structured(self):
        fields = array([(1, 1, 1)], dtype=[('i', int), ('s', '|S8'), ('f', float)])
        f1 = default_fill_value(fields)
        f2 = default_fill_value(fields.dtype)
        expected = np.array((default_fill_value(0), default_fill_value('0'), default_fill_value(0.0)), dtype=fields.dtype)
        assert_equal(f1, expected)
        assert_equal(f2, expected)

    def test_default_fill_value_void(self):
        dt = np.dtype([('v', 'V7')])
        f = default_fill_value(dt)
        assert_equal(f['v'], np.array(default_fill_value(dt['v']), dt['v']))

    def test_fillvalue(self):
        data = masked_array([1, 2, 3], fill_value=-999)
        series = data[[0, 2, 1]]
        assert_equal(series._fill_value, data._fill_value)
        mtype = [('f', float), ('s', '|S3')]
        x = array([(1, 'a'), (2, 'b'), (pi, 'pi')], dtype=mtype)
        x.fill_value = 999
        assert_equal(x.fill_value.item(), [999.0, b'999'])
        assert_equal(x['f'].fill_value, 999)
        assert_equal(x['s'].fill_value, b'999')
        x.fill_value = (9, '???')
        assert_equal(x.fill_value.item(), (9, b'???'))
        assert_equal(x['f'].fill_value, 9)
        assert_equal(x['s'].fill_value, b'???')
        x = array([1, 2, 3.1])
        x.fill_value = 999
        assert_equal(np.asarray(x.fill_value).dtype, float)
        assert_equal(x.fill_value, 999.0)
        assert_equal(x._fill_value, np.array(999.0))

    def test_subarray_fillvalue(self):
        fields = array([(1, 1, 1)], dtype=[('i', int), ('s', '|S8'), ('f', float)])
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, 'Numpy has detected')
            subfields = fields[['i', 'f']]
            assert_equal(tuple(subfields.fill_value), (999999, 1e+20))
            subfields[1:] == subfields[:-1]

    def test_fillvalue_exotic_dtype(self):
        _check_fill_value = np.ma.core._check_fill_value
        ndtype = [('i', int), ('s', '|S8'), ('f', float)]
        control = np.array((default_fill_value(0), default_fill_value('0'), default_fill_value(0.0)), dtype=ndtype)
        assert_equal(_check_fill_value(None, ndtype), control)
        ndtype = [('f0', float, (2, 2))]
        control = np.array((default_fill_value(0.0),), dtype=[('f0', float)]).astype(ndtype)
        assert_equal(_check_fill_value(None, ndtype), control)
        control = np.array((0,), dtype=[('f0', float)]).astype(ndtype)
        assert_equal(_check_fill_value(0, ndtype), control)
        ndtype = np.dtype('int, (2,3)float, float')
        control = np.array((default_fill_value(0), default_fill_value(0.0), default_fill_value(0.0)), dtype='int, float, float').astype(ndtype)
        test = _check_fill_value(None, ndtype)
        assert_equal(test, control)
        control = np.array((0, 0, 0), dtype='int, float, float').astype(ndtype)
        assert_equal(_check_fill_value(0, ndtype), control)
        M = masked_array(control)
        assert_equal(M['f1'].fill_value.ndim, 0)

    def test_fillvalue_datetime_timedelta(self):
        for timecode in ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'):
            control = numpy.datetime64('NaT', timecode)
            test = default_fill_value(numpy.dtype('<M8[' + timecode + ']'))
            np.testing.assert_equal(test, control)
            control = numpy.timedelta64('NaT', timecode)
            test = default_fill_value(numpy.dtype('<m8[' + timecode + ']'))
            np.testing.assert_equal(test, control)

    def test_extremum_fill_value(self):
        a = array([(1, (2, 3)), (4, (5, 6))], dtype=[('A', int), ('B', [('BA', int), ('BB', int)])])
        test = a.fill_value
        assert_equal(test.dtype, a.dtype)
        assert_equal(test['A'], default_fill_value(a['A']))
        assert_equal(test['B']['BA'], default_fill_value(a['B']['BA']))
        assert_equal(test['B']['BB'], default_fill_value(a['B']['BB']))
        test = minimum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], minimum_fill_value(a['A']))
        assert_equal(test[1][0], minimum_fill_value(a['B']['BA']))
        assert_equal(test[1][1], minimum_fill_value(a['B']['BB']))
        assert_equal(test[1], minimum_fill_value(a['B']))
        test = maximum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], maximum_fill_value(a['A']))
        assert_equal(test[1][0], maximum_fill_value(a['B']['BA']))
        assert_equal(test[1][1], maximum_fill_value(a['B']['BB']))
        assert_equal(test[1], maximum_fill_value(a['B']))

    def test_extremum_fill_value_subdtype(self):
        a = array(([2, 3, 4],), dtype=[('value', np.int8, 3)])
        test = minimum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], np.full(3, minimum_fill_value(a['value'])))
        test = maximum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], np.full(3, maximum_fill_value(a['value'])))

    def test_fillvalue_individual_fields(self):
        ndtype = [('a', int), ('b', int)]
        a = array(list(zip([1, 2, 3], [4, 5, 6])), fill_value=(-999, -999), dtype=ndtype)
        aa = a['a']
        aa.set_fill_value(10)
        assert_equal(aa._fill_value, np.array(10))
        assert_equal(tuple(a.fill_value), (10, -999))
        a.fill_value['b'] = -10
        assert_equal(tuple(a.fill_value), (10, -10))
        t = array(list(zip([1, 2, 3], [4, 5, 6])), dtype=ndtype)
        tt = t['a']
        tt.set_fill_value(10)
        assert_equal(tt._fill_value, np.array(10))
        assert_equal(tuple(t.fill_value), (10, default_fill_value(0)))

    def test_fillvalue_implicit_structured_array(self):
        ndtype = ('b', float)
        adtype = ('a', float)
        a = array([(1.0,), (2.0,)], mask=[(False,), (False,)], fill_value=(np.nan,), dtype=np.dtype([adtype]))
        b = empty(a.shape, dtype=[adtype, ndtype])
        b['a'] = a['a']
        b['a'].set_fill_value(a['a'].fill_value)
        f = b._fill_value[()]
        assert_(np.isnan(f[0]))
        assert_equal(f[-1], default_fill_value(1.0))

    def test_fillvalue_as_arguments(self):
        a = empty(3, fill_value=999.0)
        assert_equal(a.fill_value, 999.0)
        a = ones(3, fill_value=999.0, dtype=float)
        assert_equal(a.fill_value, 999.0)
        a = zeros(3, fill_value=0.0, dtype=complex)
        assert_equal(a.fill_value, 0.0)
        a = identity(3, fill_value=0.0, dtype=complex)
        assert_equal(a.fill_value, 0.0)

    def test_shape_argument(self):
        a = empty(shape=(3,))
        assert_equal(a.shape, (3,))
        a = ones(shape=(3,), dtype=float)
        assert_equal(a.shape, (3,))
        a = zeros(shape=(3,), dtype=complex)
        assert_equal(a.shape, (3,))

    def test_fillvalue_in_view(self):
        x = array([1, 2, 3], fill_value=1, dtype=np.int64)
        y = x.view()
        assert_(y.fill_value == 1)
        y = x.view(MaskedArray)
        assert_(y.fill_value == 1)
        y = x.view(type=MaskedArray)
        assert_(y.fill_value == 1)
        y = x.view(np.ndarray)
        y = x.view(type=np.ndarray)
        y = x.view(MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)
        y = x.view(type=MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)
        y = x.view(dtype=np.int32)
        assert_(y.fill_value == 999999)

    def test_fillvalue_bytes_or_str(self):
        a = empty(shape=(3,), dtype='(2)3S,(2)3U')
        assert_equal(a['f0'].fill_value, default_fill_value(b'spam'))
        assert_equal(a['f1'].fill_value, default_fill_value('eggs'))
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
class TestMaskedArrayFunctions:

    def setup_method(self):
        x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        xm.set_fill_value(1e+20)
        self.info = (xm, ym)

    def test_masked_where_bool(self):
        x = [1, 2]
        y = masked_where(False, x)
        assert_equal(y, [1, 2])
        assert_equal(y[1], 2)

    def test_masked_equal_wlist(self):
        x = [1, 2, 3]
        mx = masked_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [0, 0, 1])
        mx = masked_not_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [1, 1, 0])

    def test_masked_equal_fill_value(self):
        x = [1, 2, 3]
        mx = masked_equal(x, 3)
        assert_equal(mx._mask, [0, 0, 1])
        assert_equal(mx.fill_value, 3)

    def test_masked_where_condition(self):
        x = array([1.0, 2.0, 3.0, 4.0, 5.0])
        x[2] = masked
        assert_equal(masked_where(greater(x, 2), x), masked_greater(x, 2))
        assert_equal(masked_where(greater_equal(x, 2), x), masked_greater_equal(x, 2))
        assert_equal(masked_where(less(x, 2), x), masked_less(x, 2))
        assert_equal(masked_where(less_equal(x, 2), x), masked_less_equal(x, 2))
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        assert_equal(masked_where(equal(x, 2), x), masked_equal(x, 2))
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        assert_equal(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]), [99, 99, 3, 4, 5])

    def test_masked_where_oddities(self):
        atest = ones((10, 10, 10), dtype=float)
        btest = zeros(atest.shape, MaskType)
        ctest = masked_where(btest, atest)
        assert_equal(atest, ctest)

    def test_masked_where_shape_constraint(self):
        a = arange(10)
        with assert_raises(IndexError):
            masked_equal(1, a)
        test = masked_equal(a, 1)
        assert_equal(test.mask, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_masked_where_structured(self):
        a = np.zeros(10, dtype=[('A', '<f2'), ('B', '<f4')])
        with np.errstate(over='ignore'):
            am = np.ma.masked_where(a['A'] < 5, a)
        assert_equal(am.mask.dtype.names, am.dtype.names)
        assert_equal(am['A'], np.ma.masked_array(np.zeros(10), np.ones(10)))

    def test_masked_where_mismatch(self):
        x = np.arange(10)
        y = np.arange(5)
        assert_raises(IndexError, np.ma.masked_where, y > 6, x)

    def test_masked_otherfunctions(self):
        assert_equal(masked_inside(list(range(5)), 1, 3), [0, 199, 199, 199, 4])
        assert_equal(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199])
        assert_equal(masked_inside(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 1, 3).mask, [1, 1, 1, 1, 0])
        assert_equal(masked_outside(array(list(range(5)), mask=[0, 1, 0, 0, 0]), 1, 3).mask, [1, 1, 0, 0, 1])
        assert_equal(masked_equal(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 0])
        assert_equal(masked_not_equal(array([2, 2, 1, 2, 1], mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 1])

    def test_round(self):
        a = array([1.23456, 2.34567, 3.45678, 4.56789, 5.6789], mask=[0, 1, 0, 0, 0])
        assert_equal(a.round(), [1.0, 2.0, 3.0, 5.0, 6.0])
        assert_equal(a.round(1), [1.2, 2.3, 3.5, 4.6, 5.7])
        assert_equal(a.round(3), [1.235, 2.346, 3.457, 4.568, 5.679])
        b = empty_like(a)
        a.round(out=b)
        assert_equal(b, [1.0, 2.0, 3.0, 5.0, 6.0])
        x = array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)

    def test_round_with_output(self):
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked
        output = np.empty((3, 4), dtype=float)
        output.fill(-9999)
        result = np.round(xm, decimals=2, out=output)
        assert_(result is output)
        assert_equal(result, xm.round(decimals=2, out=output))
        output = empty((3, 4), dtype=float)
        result = xm.round(decimals=2, out=output)
        assert_(result is output)

    def test_round_with_scalar(self):
        a = array(1.1, mask=[False])
        assert_equal(a.round(), 1)
        a = array(1.1, mask=[True])
        assert_(a.round() is masked)
        a = array(1.1, mask=[False])
        output = np.empty(1, dtype=float)
        output.fill(-9999)
        a.round(out=output)
        assert_equal(output, 1)
        a = array(1.1, mask=[False])
        output = array(-9999.0, mask=[True])
        a.round(out=output)
        assert_equal(output[()], 1)
        a = array(1.1, mask=[True])
        output = array(-9999.0, mask=[False])
        a.round(out=output)
        assert_(output[()] is masked)

    def test_identity(self):
        a = identity(5)
        assert_(isinstance(a, MaskedArray))
        assert_equal(a, np.identity(5))

    def test_power(self):
        x = -1.1
        assert_almost_equal(power(x, 2.0), 1.21)
        assert_(power(x, masked) is masked)
        x = array([-1.1, -1.1, 1.1, 1.1, 0.0])
        b = array([0.5, 2.0, 0.5, 2.0, -1.0], mask=[0, 0, 0, 0, 1])
        y = power(x, b)
        assert_almost_equal(y, [0, 1.21, 1.04880884817, 1.21, 0.0])
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        b.mask = nomask
        y = power(x, b)
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        z = x ** b
        assert_equal(z._mask, y._mask)
        assert_almost_equal(z, y)
        assert_almost_equal(z._data, y._data)
        x **= b
        assert_equal(x._mask, y._mask)
        assert_almost_equal(x, y)
        assert_almost_equal(x._data, y._data)

    def test_power_with_broadcasting(self):
        a2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        a2m = array(a2, mask=[[1, 0, 0], [0, 0, 1]])
        b1 = np.array([2, 4, 3])
        b2 = np.array([b1, b1])
        b2m = array(b2, mask=[[0, 1, 0], [0, 1, 0]])
        ctrl = array([[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]], mask=[[1, 1, 0], [0, 1, 1]])
        test = a2m ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        test = a2m ** b2
        assert_equal(test, ctrl)
        assert_equal(test.mask, a2m.mask)
        test = a2 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, b2m.mask)
        ctrl = array([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]], mask=[[0, 1, 0], [0, 1, 0]])
        test = b1 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        test = b2m ** b1
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_where(self):
        x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        xm.set_fill_value(1e+20)
        d = where(xm > 2, xm, -9)
        assert_equal(d, [-9.0, -9.0, -9.0, -9.0, -9.0, 4.0, -9.0, -9.0, 10.0, -9.0, -9.0, 3.0])
        assert_equal(d._mask, xm._mask)
        d = where(xm > 2, -9, ym)
        assert_equal(d, [5.0, 0.0, 3.0, 2.0, -1.0, -9.0, -9.0, -10.0, -9.0, 1.0, 0.0, -9.0])
        assert_equal(d._mask, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        d = where(xm > 2, xm, masked)
        assert_equal(d, [-9.0, -9.0, -9.0, -9.0, -9.0, 4.0, -9.0, -9.0, 10.0, -9.0, -9.0, 3.0])
        tmp = xm._mask.copy()
        tmp[(xm <= 2).filled(True)] = True
        assert_equal(d._mask, tmp)
        with np.errstate(invalid='warn'):
            with pytest.warns(RuntimeWarning, match='invalid value'):
                ixm = xm.astype(int)
        d = where(ixm > 2, ixm, masked)
        assert_equal(d, [-9, -9, -9, -9, -9, 4, -9, -9, 10, -9, -9, 3])
        assert_equal(d.dtype, ixm.dtype)

    def test_where_object(self):
        a = np.array(None)
        b = masked_array(None)
        r = b.copy()
        assert_equal(np.ma.where(True, a, a), r)
        assert_equal(np.ma.where(True, b, b), r)

    def test_where_with_masked_choice(self):
        x = arange(10)
        x[3] = masked
        c = x >= 8
        z = where(c, x, masked)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is masked)
        assert_(z[7] is masked)
        assert_(z[8] is not masked)
        assert_(z[9] is not masked)
        assert_equal(x, z)
        z = where(c, masked, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)

    def test_where_with_masked_condition(self):
        x = array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)
        x = arange(1, 6)
        x[-1] = masked
        y = arange(1, 6) * 10
        y[2] = masked
        c = array([1, 1, 1, 0, 0], mask=[1, 0, 0, 0, 0])
        cm = c.filled(1)
        z = where(c, x, y)
        zm = where(cm, x, y)
        assert_equal(z, zm)
        assert_(getmask(zm) is nomask)
        assert_equal(zm, [1, 2, 3, 40, 50])
        z = where(c, masked, 1)
        assert_equal(z, [99, 99, 99, 1, 1])
        z = where(c, 1, masked)
        assert_equal(z, [99, 1, 1, 99, 99])

    def test_where_type(self):
        x = np.arange(4, dtype=np.int32)
        y = np.arange(4, dtype=np.float32) * 2.2
        test = where(x > 1.5, y, x).dtype
        control = np.result_type(np.int32, np.float32)
        assert_equal(test, control)

    def test_where_broadcast(self):
        x = np.arange(9).reshape(3, 3)
        y = np.zeros(3)
        core = np.where([1, 0, 1], x, y)
        ma = where([1, 0, 1], x, y)
        assert_equal(core, ma)
        assert_equal(core.dtype, ma.dtype)

    def test_where_structured(self):
        dt = np.dtype([('a', int), ('b', int)])
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
        y = np.array((10, 20), dtype=dt)
        core = np.where([0, 1, 1], x, y)
        ma = np.where([0, 1, 1], x, y)
        assert_equal(core, ma)
        assert_equal(core.dtype, ma.dtype)

    def test_where_structured_masked(self):
        dt = np.dtype([('a', int), ('b', int)])
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
        ma = where([0, 1, 1], x, masked)
        expected = masked_where([1, 0, 0], x)
        assert_equal(ma.dtype, expected.dtype)
        assert_equal(ma, expected)
        assert_equal(ma.mask, expected.mask)

    def test_masked_invalid_error(self):
        a = np.arange(5, dtype=object)
        a[3] = np.PINF
        a[2] = np.NaN
        with pytest.raises(TypeError, match='not supported for the input types'):
            np.ma.masked_invalid(a)

    def test_masked_invalid_pandas(self):

        class Series:
            _data = 'nonsense'

            def __array__(self):
                return np.array([5, np.nan, np.inf])
        arr = np.ma.masked_invalid(Series())
        assert_array_equal(arr._data, np.array(Series()))
        assert_array_equal(arr._mask, [False, True, True])

    @pytest.mark.parametrize('copy', [True, False])
    def test_masked_invalid_full_mask(self, copy):
        a = np.ma.array([1, 2, 3, 4])
        assert a._mask is nomask
        res = np.ma.masked_invalid(a, copy=copy)
        assert res.mask is not nomask
        assert a.mask is nomask
        assert np.may_share_memory(a._data, res._data) != copy

    def test_choose(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        chosen = choose([2, 3, 1, 0], choices)
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='clip')
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='wrap')
        assert_equal(chosen, array([20, 1, 12, 3]))
        indices_ = array([2, 4, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([99, 1, 12, 99]))
        assert_equal(chosen.mask, [1, 0, 0, 1])
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([20, 31, 12, 3]))
        assert_equal(chosen.mask, [1, 0, 0, 1])

    def test_choose_with_out(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        store = empty(4, dtype=int)
        chosen = choose([2, 3, 1, 0], choices, out=store)
        assert_equal(store, array([20, 31, 12, 3]))
        assert_(store is chosen)
        store = empty(4, dtype=int)
        indices_ = array([2, 3, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([99, 31, 12, 99]))
        assert_equal(store.mask, [1, 0, 0, 1])
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        store = empty(4, dtype=int).view(ndarray)
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([999999, 31, 12, 999999]))

    def test_reshape(self):
        a = arange(10)
        a[0] = masked
        b = a.reshape((5, 2))
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        b = a.reshape(5, 2)
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        b = a.reshape((5, 2), order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])
        b = a.reshape(5, 2, order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])
        c = np.reshape(a, (2, 5))
        assert_(isinstance(c, MaskedArray))
        assert_equal(c.shape, (2, 5))
        assert_(c[0, 0] is masked)
        assert_(c.flags['C'])

    def test_make_mask_descr(self):
        ntype = [('a', float), ('b', float)]
        test = make_mask_descr(ntype)
        assert_equal(test, [('a', bool), ('b', bool)])
        assert_(test is make_mask_descr(test))
        ntype = (float, 2)
        test = make_mask_descr(ntype)
        assert_equal(test, (bool, 2))
        assert_(test is make_mask_descr(test))
        ntype = float
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype(bool))
        assert_(test is make_mask_descr(test))
        ntype = [('a', float), ('b', [('ba', float), ('bb', float)])]
        test = make_mask_descr(ntype)
        control = np.dtype([('a', 'b1'), ('b', [('ba', 'b1'), ('bb', 'b1')])])
        assert_equal(test, control)
        assert_(test is make_mask_descr(test))
        ntype = [('a', (float, 2))]
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype([('a', (bool, 2))]))
        assert_(test is make_mask_descr(test))
        ntype = [(('A', 'a'), float)]
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype([(('A', 'a'), bool)]))
        assert_(test is make_mask_descr(test))
        base_type = np.dtype([('a', int, 3)])
        base_mtype = make_mask_descr(base_type)
        sub_type = np.dtype([('a', int), ('b', base_mtype)])
        test = make_mask_descr(sub_type)
        assert_equal(test, np.dtype([('a', bool), ('b', [('a', bool, 3)])]))
        assert_(test.fields['b'][0] is base_mtype)

    def test_make_mask(self):
        mask = [0, 1]
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        mask = np.array([0, 1], dtype=bool)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [1, 1])
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, mdtype)
        assert_equal(test, mask)
        mdtype = [('a', float), ('b', float)]
        bdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, bdtype)
        assert_equal(test, np.array([(0, 0), (0, 1)], dtype=bdtype))
        mask = np.array((False, True), dtype='?,?')[()]
        assert_(isinstance(mask, np.void))
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test, mask)
        assert_(test is not mask)
        mask = np.array((0, 1), dtype='i4,i4')[()]
        test2 = make_mask(mask, dtype=mask.dtype)
        assert_equal(test2, test)
        bools = [True, False]
        dtypes = [MaskType, float]
        msgformat = 'copy=%s, shrink=%s, dtype=%s'
        for cpy, shr, dt in itertools.product(bools, bools, dtypes):
            res = make_mask(nomask, copy=cpy, shrink=shr, dtype=dt)
            assert_(res is nomask, msgformat % (cpy, shr, dt))

    def test_mask_or(self):
        mtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1), (1, 0), (0, 0)], dtype=mtype)
        test = mask_or(mask, nomask)
        assert_equal(test, mask)
        test = mask_or(nomask, mask)
        assert_equal(test, mask)
        test = mask_or(mask, False)
        assert_equal(test, mask)
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=mtype)
        test = mask_or(mask, other)
        control = np.array([(0, 1), (0, 1), (1, 1), (0, 1)], dtype=mtype)
        assert_equal(test, control)
        othertype = [('A', bool), ('B', bool)]
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=othertype)
        try:
            test = mask_or(mask, other)
        except ValueError:
            pass
        dtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        amask = np.array([(0, (1, 0)), (0, (1, 0))], dtype=dtype)
        bmask = np.array([(1, (0, 1)), (0, (0, 0))], dtype=dtype)
        cntrl = np.array([(1, (1, 1)), (0, (1, 0))], dtype=dtype)
        assert_equal(mask_or(amask, bmask), cntrl)

    def test_flatten_mask(self):
        mask = np.array([0, 0, 1], dtype=bool)
        assert_equal(flatten_mask(mask), mask)
        mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
        test = flatten_mask(mask)
        control = np.array([0, 0, 0, 1], dtype=bool)
        assert_equal(test, control)
        mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        data = [(0, (0, 0)), (0, (0, 1))]
        mask = np.array(data, dtype=mdtype)
        test = flatten_mask(mask)
        control = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        assert_equal(test, control)

    def test_on_ndarray(self):
        a = np.array([1, 2, 3, 4])
        m = array(a, mask=False)
        test = anom(a)
        assert_equal(test, m.anom())
        test = reshape(a, (2, 2))
        assert_equal(test, m.reshape(2, 2))

    def test_compress(self):
        arr = np.arange(8)
        arr.shape = (4, 2)
        cond = np.array([True, False, True, True])
        control = arr[[0, 2, 3]]
        test = np.ma.compress(cond, arr, axis=0)
        assert_equal(test, control)
        marr = np.ma.array(arr)
        test = np.ma.compress(cond, marr, axis=0)
        assert_equal(test, control)

    def test_compressed(self):
        a = np.ma.array([1, 2])
        test = np.ma.compressed(a)
        assert_(type(test) is np.ndarray)

        class A(np.ndarray):
            pass
        a = np.ma.array(A(shape=0))
        test = np.ma.compressed(a)
        assert_(type(test) is A)
        test = np.ma.compressed([[1], [2]])
        assert_equal(test.ndim, 1)
        test = np.ma.compressed([[[[[1]]]]])
        assert_equal(test.ndim, 1)

        class M(MaskedArray):
            pass
        test = np.ma.compressed(M([[[]], [[]]]))
        assert_equal(test.ndim, 1)

        class M(MaskedArray):

            def compressed(self):
                return 42
        test = np.ma.compressed(M([[[]], [[]]]))
        assert_equal(test, 42)

    def test_convolve(self):
        a = masked_equal(np.arange(5), 2)
        b = np.array([1, 1])
        test = np.ma.convolve(a, b)
        assert_equal(test, masked_equal([0, 1, -1, -1, 7, 4], -1))
        test = np.ma.convolve(a, b, propagate_mask=False)
        assert_equal(test, masked_equal([0, 1, 1, 3, 7, 4], -1))
        test = np.ma.convolve([1, 1], [1, 1, 1])
        assert_equal(test, masked_equal([1, 2, 2, 1], -1))
        a = [1, 1]
        b = masked_equal([1, -1, -1, 1], -1)
        test = np.ma.convolve(a, b, propagate_mask=False)
        assert_equal(test, masked_equal([1, 1, -1, 1, 1], -1))
        test = np.ma.convolve(a, b, propagate_mask=True)
        assert_equal(test, masked_equal([-1, -1, -1, -1, -1], -1))
from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestBinop:

    def test_inplace(self):
        assert_array_almost_equal(np.array([0.5]) * np.array([1.0, 2.0]), [0.5, 1.0])
        d = np.array([0.5, 0.5])[::2]
        assert_array_almost_equal(d * (d * np.array([1.0, 2.0])), [0.25, 0.5])
        a = np.array([0.5])
        b = np.array([0.5])
        c = a + b
        c = a - b
        c = a * b
        c = a / b
        assert_equal(a, b)
        assert_almost_equal(c, 1.0)
        c = a + b * 2.0 / b * a - a / b
        assert_equal(a, b)
        assert_equal(c, 0.5)
        a = np.array([5])
        b = np.array([3])
        c = a * a / b
        assert_almost_equal(c, 25 / 3)
        assert_equal(a, 5)
        assert_equal(b, 3)

    @pytest.mark.xfail(IS_PYPY, reason='Bug in pypy3.{9, 10}-v7.3.13, #24862')
    def test_ufunc_binop_interaction(self):
        ops = {'add': (np.add, True, float), 'sub': (np.subtract, True, float), 'mul': (np.multiply, True, float), 'truediv': (np.true_divide, True, float), 'floordiv': (np.floor_divide, True, float), 'mod': (np.remainder, True, float), 'divmod': (np.divmod, False, float), 'pow': (np.power, True, int), 'lshift': (np.left_shift, True, int), 'rshift': (np.right_shift, True, int), 'and': (np.bitwise_and, True, int), 'xor': (np.bitwise_xor, True, int), 'or': (np.bitwise_or, True, int), 'matmul': (np.matmul, True, float)}

        class Coerced(Exception):
            pass

        def array_impl(self):
            raise Coerced

        def op_impl(self, other):
            return 'forward'

        def rop_impl(self, other):
            return 'reverse'

        def iop_impl(self, other):
            return 'in-place'

        def array_ufunc_impl(self, ufunc, method, *args, **kwargs):
            return ('__array_ufunc__', ufunc, method, args, kwargs)

        def make_obj(base, array_priority=False, array_ufunc=False, alleged_module='__main__'):
            class_namespace = {'__array__': array_impl}
            if array_priority is not False:
                class_namespace['__array_priority__'] = array_priority
            for op in ops:
                class_namespace['__{0}__'.format(op)] = op_impl
                class_namespace['__r{0}__'.format(op)] = rop_impl
                class_namespace['__i{0}__'.format(op)] = iop_impl
            if array_ufunc is not False:
                class_namespace['__array_ufunc__'] = array_ufunc
            eval_namespace = {'base': base, 'class_namespace': class_namespace, '__name__': alleged_module}
            MyType = eval("type('MyType', (base,), class_namespace)", eval_namespace)
            if issubclass(MyType, np.ndarray):
                return np.arange(3, 7).reshape(2, 2).view(MyType)
            else:
                return MyType()

        def check(obj, binop_override_expected, ufunc_override_expected, inplace_override_expected, check_scalar=True):
            for op, (ufunc, has_inplace, dtype) in ops.items():
                err_msg = 'op: %s, ufunc: %s, has_inplace: %s, dtype: %s' % (op, ufunc, has_inplace, dtype)
                check_objs = [np.arange(3, 7, dtype=dtype).reshape(2, 2)]
                if check_scalar:
                    check_objs.append(check_objs[0][0])
                for arr in check_objs:
                    arr_method = getattr(arr, '__{0}__'.format(op))

                    def first_out_arg(result):
                        if op == 'divmod':
                            assert_(isinstance(result, tuple))
                            return result[0]
                        else:
                            return result
                    if binop_override_expected:
                        assert_equal(arr_method(obj), NotImplemented, err_msg)
                    elif ufunc_override_expected:
                        assert_equal(arr_method(obj)[0], '__array_ufunc__', err_msg)
                    elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                        res = first_out_arg(arr_method(obj))
                        assert_(res.__class__ is obj.__class__, err_msg)
                    else:
                        assert_raises((TypeError, Coerced), arr_method, obj, err_msg=err_msg)
                    arr_rmethod = getattr(arr, '__r{0}__'.format(op))
                    if ufunc_override_expected:
                        res = arr_rmethod(obj)
                        assert_equal(res[0], '__array_ufunc__', err_msg=err_msg)
                        assert_equal(res[1], ufunc, err_msg=err_msg)
                    elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                        res = first_out_arg(arr_rmethod(obj))
                        assert_(res.__class__ is obj.__class__, err_msg)
                    else:
                        assert_raises((TypeError, Coerced), arr_rmethod, obj, err_msg=err_msg)
                    if has_inplace and isinstance(arr, np.ndarray):
                        arr_imethod = getattr(arr, '__i{0}__'.format(op))
                        if inplace_override_expected:
                            assert_equal(arr_method(obj), NotImplemented, err_msg=err_msg)
                        elif ufunc_override_expected:
                            res = arr_imethod(obj)
                            assert_equal(res[0], '__array_ufunc__', err_msg)
                            assert_equal(res[1], ufunc, err_msg)
                            assert_(type(res[-1]['out']) is tuple, err_msg)
                            assert_(res[-1]['out'][0] is arr, err_msg)
                        elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                            assert_(arr_imethod(obj) is arr, err_msg)
                        else:
                            assert_raises((TypeError, Coerced), arr_imethod, obj, err_msg=err_msg)
                    op_fn = getattr(operator, op, None)
                    if op_fn is None:
                        op_fn = getattr(operator, op + '_', None)
                    if op_fn is None:
                        op_fn = getattr(builtins, op)
                    assert_equal(op_fn(obj, arr), 'forward', err_msg)
                    if not isinstance(obj, np.ndarray):
                        if binop_override_expected:
                            assert_equal(op_fn(arr, obj), 'reverse', err_msg)
                        elif ufunc_override_expected:
                            assert_equal(op_fn(arr, obj)[0], '__array_ufunc__', err_msg)
                    if ufunc_override_expected:
                        assert_equal(ufunc(obj, arr)[0], '__array_ufunc__', err_msg)
        check(make_obj(object), False, False, False)
        check(make_obj(object, array_priority=-2 ** 30), False, False, False)
        check(make_obj(object, array_priority=1), True, False, True)
        check(make_obj(np.ndarray, array_priority=1), False, False, False, check_scalar=False)
        check(make_obj(object, array_priority=1, array_ufunc=array_ufunc_impl), False, True, False)
        check(make_obj(np.ndarray, array_priority=1, array_ufunc=array_ufunc_impl), False, True, False)
        check(make_obj(object, array_ufunc=None), True, False, False)
        check(make_obj(np.ndarray, array_ufunc=None), True, False, False, check_scalar=False)

    @pytest.mark.parametrize('priority', [None, 'runtime error'])
    def test_ufunc_binop_bad_array_priority(self, priority):

        class BadPriority:

            @property
            def __array_priority__(self):
                if priority == 'runtime error':
                    raise RuntimeError('RuntimeError in __array_priority__!')
                return priority

            def __radd__(self, other):
                return 'result'

        class LowPriority(np.ndarray):
            __array_priority__ = -1000
        res = np.arange(3).view(LowPriority) + BadPriority()
        assert res.shape == (3,)
        assert res[0] == 'result'

    def test_ufunc_override_normalize_signature(self):

        class SomeClass:

            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                return kw
        a = SomeClass()
        kw = np.add(a, [1])
        assert_('sig' not in kw and 'signature' not in kw)
        kw = np.add(a, [1], sig='ii->i')
        assert_('sig' not in kw and 'signature' in kw)
        assert_equal(kw['signature'], 'ii->i')
        kw = np.add(a, [1], signature='ii->i')
        assert_('sig' not in kw and 'signature' in kw)
        assert_equal(kw['signature'], 'ii->i')

    def test_array_ufunc_index(self):

        class CheckIndex:

            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                for i, a in enumerate(inputs):
                    if a is self:
                        return i
                for j, a in enumerate(kw['out']):
                    if a is self:
                        return (j,)
        a = CheckIndex()
        dummy = np.arange(2.0)
        assert_equal(np.sin(a), 0)
        assert_equal(np.sin(dummy, a), (0,))
        assert_equal(np.sin(dummy, out=a), (0,))
        assert_equal(np.sin(dummy, out=(a,)), (0,))
        assert_equal(np.sin(a, a), 0)
        assert_equal(np.sin(a, out=a), 0)
        assert_equal(np.sin(a, out=(a,)), 0)
        assert_equal(np.modf(dummy, a), (0,))
        assert_equal(np.modf(dummy, None, a), (1,))
        assert_equal(np.modf(dummy, dummy, a), (1,))
        assert_equal(np.modf(dummy, out=(a, None)), (0,))
        assert_equal(np.modf(dummy, out=(a, dummy)), (0,))
        assert_equal(np.modf(dummy, out=(None, a)), (1,))
        assert_equal(np.modf(dummy, out=(dummy, a)), (1,))
        assert_equal(np.modf(a, out=(dummy, a)), 0)
        with assert_raises(TypeError):
            np.modf(dummy, out=a)
        assert_raises(ValueError, np.modf, dummy, out=(a,))
        assert_equal(np.add(a, dummy), 0)
        assert_equal(np.add(dummy, a), 1)
        assert_equal(np.add(dummy, dummy, a), (0,))
        assert_equal(np.add(dummy, a, a), 1)
        assert_equal(np.add(dummy, dummy, out=a), (0,))
        assert_equal(np.add(dummy, dummy, out=(a,)), (0,))
        assert_equal(np.add(a, dummy, out=a), 0)

    def test_out_override(self):

        class OutClass(np.ndarray):

            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                if 'out' in kw:
                    tmp_kw = kw.copy()
                    tmp_kw.pop('out')
                    func = getattr(ufunc, method)
                    kw['out'][0][...] = func(*inputs, **tmp_kw)
        A = np.array([0]).view(OutClass)
        B = np.array([5])
        C = np.array([6])
        np.multiply(C, B, A)
        assert_equal(A[0], 30)
        assert_(isinstance(A, OutClass))
        A[0] = 0
        np.multiply(C, B, out=A)
        assert_equal(A[0], 30)
        assert_(isinstance(A, OutClass))

    def test_pow_override_with_errors(self):

        class PowerOnly(np.ndarray):

            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                if ufunc is not np.power:
                    raise NotImplementedError
                return 'POWER!'
        a = np.array(5.0, dtype=np.float64).view(PowerOnly)
        assert_equal(a ** 2.5, 'POWER!')
        with assert_raises(NotImplementedError):
            a ** 0.5
        with assert_raises(NotImplementedError):
            a ** 0
        with assert_raises(NotImplementedError):
            a ** 1
        with assert_raises(NotImplementedError):
            a ** (-1)
        with assert_raises(NotImplementedError):
            a ** 2

    def test_pow_array_object_dtype(self):

        class SomeClass:

            def __init__(self, num=None):
                self.num = num

            def __mul__(self, other):
                raise AssertionError('__mul__ should not be called')

            def __div__(self, other):
                raise AssertionError('__div__ should not be called')

            def __pow__(self, exp):
                return SomeClass(num=self.num ** exp)

            def __eq__(self, other):
                if isinstance(other, SomeClass):
                    return self.num == other.num
            __rpow__ = __pow__

        def pow_for(exp, arr):
            return np.array([x ** exp for x in arr])
        obj_arr = np.array([SomeClass(1), SomeClass(2), SomeClass(3)])
        assert_equal(obj_arr ** 0.5, pow_for(0.5, obj_arr))
        assert_equal(obj_arr ** 0, pow_for(0, obj_arr))
        assert_equal(obj_arr ** 1, pow_for(1, obj_arr))
        assert_equal(obj_arr ** (-1), pow_for(-1, obj_arr))
        assert_equal(obj_arr ** 2, pow_for(2, obj_arr))

    def test_pos_array_ufunc_override(self):

        class A(np.ndarray):

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return getattr(ufunc, method)(*[i.view(np.ndarray) for i in inputs], **kwargs)
        tst = np.array('foo').view(A)
        with assert_raises(TypeError):
            +tst
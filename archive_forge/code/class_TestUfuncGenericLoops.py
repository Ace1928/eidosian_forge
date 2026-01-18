import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
class TestUfuncGenericLoops:
    """Test generic loops.

    The loops to be tested are:

        PyUFunc_ff_f_As_dd_d
        PyUFunc_ff_f
        PyUFunc_dd_d
        PyUFunc_gg_g
        PyUFunc_FF_F_As_DD_D
        PyUFunc_DD_D
        PyUFunc_FF_F
        PyUFunc_GG_G
        PyUFunc_OO_O
        PyUFunc_OO_O_method
        PyUFunc_f_f_As_d_d
        PyUFunc_d_d
        PyUFunc_f_f
        PyUFunc_g_g
        PyUFunc_F_F_As_D_D
        PyUFunc_F_F
        PyUFunc_D_D
        PyUFunc_G_G
        PyUFunc_O_O
        PyUFunc_O_O_method
        PyUFunc_On_Om

    Where:

        f -- float
        d -- double
        g -- long double
        F -- complex float
        D -- complex double
        G -- complex long double
        O -- python object

    It is difficult to assure that each of these loops is entered from the
    Python level as the special cased loops are a moving target and the
    corresponding types are architecture dependent. We probably need to
    define C level testing ufuncs to get at them. For the time being, I've
    just looked at the signatures registered in the build directory to find
    relevant functions.

    """
    np_dtypes = [(np.single, np.single), (np.single, np.double), (np.csingle, np.csingle), (np.csingle, np.cdouble), (np.double, np.double), (np.longdouble, np.longdouble), (np.cdouble, np.cdouble), (np.clongdouble, np.clongdouble)]

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_unary_PyUFunc(self, input_dtype, output_dtype, f=np.exp, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    def f2(x, y):
        return x ** y

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs, xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    class foo:

        def conjugate(self):
            return np.bool_(1)

        def logical_xor(self, obj):
            return np.bool_(1)

    def test_unary_PyUFunc_O_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.abs(x) == 1))

    def test_unary_PyUFunc_O_O_method_simple(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.conjugate(x) == True))

    def test_binary_PyUFunc_OO_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.add(x, x) == 2))

    def test_binary_PyUFunc_OO_O_method(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_binary_PyUFunc_On_Om_method(self, foo=foo):
        x = np.full((10, 2, 3), foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_python_complex_conjugate(self):
        arr = np.array([1 + 2j, 3 - 4j], dtype='O')
        assert isinstance(arr[0], complex)
        res = np.conjugate(arr)
        assert res.dtype == np.dtype('O')
        assert_array_equal(res, np.array([1 - 2j, 3 + 4j], dtype='O'))

    @pytest.mark.parametrize('ufunc', UNARY_OBJECT_UFUNCS)
    def test_unary_PyUFunc_O_O_method_full(self, ufunc):
        """Compare the result of the object loop with non-object one"""
        val = np.float64(np.pi / 4)

        class MyFloat(np.float64):

            def __getattr__(self, attr):
                try:
                    return super().__getattr__(attr)
                except AttributeError:
                    return lambda: getattr(np.core.umath, attr)(val)
        num_arr = np.array(val, dtype=np.float64)
        obj_arr = np.array(MyFloat(val), dtype='O')
        with np.errstate(all='raise'):
            try:
                res_num = ufunc(num_arr)
            except Exception as exc:
                with assert_raises(type(exc)):
                    ufunc(obj_arr)
            else:
                res_obj = ufunc(obj_arr)
                assert_array_almost_equal(res_num.astype('O'), res_obj)
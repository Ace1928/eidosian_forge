import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
class TestNdimageFilters:

    def _validate_complex(self, array, kernel, type2, mode='reflect', cval=0):
        real_dtype = numpy.asarray([], dtype=type2).real.dtype
        expected = _complex_correlate(array, kernel, real_dtype, convolve=False, mode=mode, cval=cval)
        if array.ndim == 1:
            correlate = functools.partial(ndimage.correlate1d, axis=-1, mode=mode, cval=cval)
            convolve = functools.partial(ndimage.convolve1d, axis=-1, mode=mode, cval=cval)
        else:
            correlate = functools.partial(ndimage.correlate, mode=mode, cval=cval)
            convolve = functools.partial(ndimage.convolve, mode=mode, cval=cval)
        output = correlate(array, kernel, output=type2)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)
        output = numpy.zeros_like(array, dtype=type2)
        correlate(array, kernel, output=output)
        assert_array_almost_equal(expected, output)
        output = convolve(array, kernel, output=type2)
        expected = _complex_correlate(array, kernel, real_dtype, convolve=True, mode=mode, cval=cval)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)
        convolve(array, kernel, output=output)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)
        with pytest.warns(UserWarning, match='promoting specified output dtype to complex'):
            correlate(array, kernel, output=real_dtype)
        with pytest.warns(UserWarning, match='promoting specified output dtype to complex'):
            convolve(array, kernel, output=real_dtype)
        output_real = numpy.zeros_like(array, dtype=real_dtype)
        with assert_raises(RuntimeError):
            correlate(array, kernel, output=output_real)
        with assert_raises(RuntimeError):
            convolve(array, kernel, output=output_real)

    def test_correlate01(self):
        array = numpy.array([1, 2])
        weights = numpy.array([2])
        expected = [2, 4]
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate01_overlap(self):
        array = numpy.arange(256).reshape(16, 16)
        weights = numpy.array([2])
        expected = 2 * array
        ndimage.correlate1d(array, weights, output=array)
        assert_array_almost_equal(array, expected)

    def test_correlate02(self):
        array = numpy.array([1, 2, 3])
        kernel = numpy.array([1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate03(self):
        array = numpy.array([1])
        weights = numpy.array([1, 1])
        expected = [2]
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate04(self):
        array = numpy.array([1, 2])
        tcor = [2, 3]
        tcov = [3, 4]
        weights = numpy.array([1, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate05(self):
        array = numpy.array([1, 2, 3])
        tcor = [2, 3, 5]
        tcov = [3, 5, 6]
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(tcov, output)

    def test_correlate06(self):
        array = numpy.array([1, 2, 3])
        tcor = [9, 14, 17]
        tcov = [7, 10, 15]
        weights = numpy.array([1, 2, 3])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate07(self):
        array = numpy.array([1, 2, 3])
        expected = [5, 8, 11]
        weights = numpy.array([1, 2, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate08(self):
        array = numpy.array([1, 2, 3])
        tcor = [1, 2, 5]
        tcov = [3, 6, 7]
        weights = numpy.array([1, 2, -1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate09(self):
        array = []
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate10(self):
        array = [[]]
        kernel = numpy.array([[1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate11(self):
        array = numpy.array([[1, 2, 3], [4, 5, 6]])
        kernel = numpy.array([[1, 1], [1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[12, 16, 18], [18, 22, 24]], output)

    def test_correlate12(self):
        array = numpy.array([[1, 2, 3], [4, 5, 6]])
        kernel = numpy.array([[1, 0], [0, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_kernel', types)
    def test_correlate13(self, dtype_array, dtype_kernel):
        kernel = numpy.array([[1, 0], [0, 1]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=dtype_kernel)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, dtype_kernel)
        output = ndimage.convolve(array, kernel, output=dtype_kernel)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, dtype_kernel)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate14(self, dtype_array, dtype_output):
        kernel = numpy.array([[1, 0], [0, 1]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = numpy.zeros(array.shape, dtype_output)
        ndimage.correlate(array, kernel, output=output)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, dtype_output)
        ndimage.convolve(array, kernel, output=output)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, dtype_output)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate15(self, dtype_array):
        kernel = numpy.array([[1, 0], [0, 1]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, numpy.float32)
        output = ndimage.convolve(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, numpy.float32)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate16(self, dtype_array):
        kernel = numpy.array([[0.5, 0], [0, 0.5]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[1, 1.5, 2.5], [2.5, 3, 4]], output)
        assert_equal(output.dtype.type, numpy.float32)
        output = ndimage.convolve(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[3, 4, 4.5], [4.5, 5.5, 6]], output)
        assert_equal(output.dtype.type, numpy.float32)

    def test_correlate17(self):
        array = numpy.array([1, 2, 3])
        tcor = [3, 5, 6]
        tcov = [2, 3, 5]
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate18(self, dtype_array):
        kernel = numpy.array([[1, 0], [0, 1]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32, mode='nearest', origin=-1)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, numpy.float32)
        output = ndimage.convolve(array, kernel, output=numpy.float32, mode='nearest', origin=-1)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, numpy.float32)

    def test_correlate_mode_sequence(self):
        kernel = numpy.ones((2, 2))
        array = numpy.ones((3, 3), float)
        with assert_raises(RuntimeError):
            ndimage.correlate(array, kernel, mode=['nearest', 'reflect'])
        with assert_raises(RuntimeError):
            ndimage.convolve(array, kernel, mode=['nearest', 'reflect'])

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate19(self, dtype_array):
        kernel = numpy.array([[1, 0], [0, 1]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32, mode='nearest', origin=[-1, 0])
        assert_array_almost_equal([[5, 6, 8], [8, 9, 11]], output)
        assert_equal(output.dtype.type, numpy.float32)
        output = ndimage.convolve(array, kernel, output=numpy.float32, mode='nearest', origin=[-1, 0])
        assert_array_almost_equal([[3, 5, 6], [6, 8, 9]], output)
        assert_equal(output.dtype.type, numpy.float32)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate20(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)

    def test_correlate21(self):
        array = numpy.array([[1, 2, 3], [2, 4, 6]])
        expected = [[5, 10, 15], [7, 14, 21]]
        weights = numpy.array([1, 2, 1])
        output = ndimage.correlate1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate22(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[6, 12, 18], [6, 12, 18]]
        array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, mode='wrap', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0, mode='wrap', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate23(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, mode='nearest', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0, mode='nearest', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate24(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        tcor = [[7, 14, 21], [8, 16, 24]]
        tcov = [[4, 8, 12], [5, 10, 15]]
        array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0, mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcov)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate25(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        tcor = [[4, 8, 12], [5, 10, 15]]
        tcov = [[7, 14, 21], [8, 16, 24]]
        array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0, mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcov)

    def test_correlate26(self):
        y = ndimage.convolve1d(numpy.ones(1), numpy.ones(5), mode='mirror')
        assert_array_equal(y, numpy.array(5.0))
        y = ndimage.correlate1d(numpy.ones(1), numpy.ones(5), mode='mirror')
        assert_array_equal(y, numpy.array(5.0))

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_kernel(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([[1, 0], [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    @pytest.mark.parametrize('mode', ['grid-constant', 'constant'])
    def test_correlate_complex_kernel_cval(self, dtype_input, dtype_kernel, dtype_output, mode):
        kernel = numpy.array([[1, 0], [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode=mode, cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    def test_correlate_complex_kernel_invalid_cval(self, dtype_input, dtype_kernel):
        kernel = numpy.array([[1, 0], [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_input)
        for func in [ndimage.convolve, ndimage.correlate, ndimage.convolve1d, ndimage.correlate1d]:
            with pytest.raises(ValueError):
                func(array, kernel, mode='constant', cval=5.0 + 1j, output=numpy.complex64)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype_kernel)
        array = numpy.array([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel_cval(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype_kernel)
        array = numpy.array([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode='constant', cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([[1, 0], [0, 1]], dtype_kernel)
        array = numpy.array([[1, 2j, 3], [1 + 4j, 5, 6j]], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([1, 0, 1], dtype_kernel)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_cval(self, dtype_input, dtype_kernel, dtype_output):
        kernel = numpy.array([1, 0, 1], dtype_kernel)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode='constant', cval=5 - 3j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel(self, dtype, dtype_output):
        kernel = numpy.array([[1, 0], [0, 1 + 1j]], dtype)
        array = numpy.array([[1, 2j, 3], [1 + 4j, 5, 6j]], dtype)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel_cval(self, dtype, dtype_output):
        kernel = numpy.array([[1, 0], [0, 1 + 1j]], dtype)
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype)
        self._validate_complex(array, kernel, dtype_output, mode='constant', cval=5.0 + 2j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel(self, dtype, dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel_cval(self, dtype, dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(array, kernel, dtype_output, mode='constant', cval=5.0 + 2j)

    def test_gauss01(self):
        input = numpy.array([[1, 2, 3], [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 0)
        assert_array_almost_equal(output, input)

    def test_gauss02(self):
        input = numpy.array([[1, 2, 3], [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 1.0)
        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

    def test_gauss03(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output = ndimage.gaussian_filter(input, [1.0, 1.0])
        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)
        assert_almost_equal(output.sum(dtype='d'), input.sum(dtype='d'), decimal=0)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss04(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss05(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0], order=1, output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss06(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        output2 = ndimage.gaussian_filter(input, 1.0, output=otype)
        assert_array_almost_equal(output1, output2)

    def test_gauss_memory_overlap(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output1 = ndimage.gaussian_filter(input, 1.0)
        ndimage.gaussian_filter(input, 1.0, output=input)
        assert_array_almost_equal(output1, input)

    @pytest.mark.parametrize(('filter_func', 'extra_args', 'size0', 'size'), [(ndimage.gaussian_filter, (), 0, 1.0), (ndimage.uniform_filter, (), 1, 3), (ndimage.minimum_filter, (), 1, 3), (ndimage.maximum_filter, (), 1, 3), (ndimage.median_filter, (), 1, 3), (ndimage.rank_filter, (1,), 1, 3), (ndimage.percentile_filter, (40,), 1, 3)])
    @pytest.mark.parametrize('axes', tuple(itertools.combinations(range(-3, 3), 1)) + tuple(itertools.combinations(range(-3, 3), 2)) + ((0, 1, 2),))
    def test_filter_axes(self, filter_func, extra_args, size0, size, axes):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
        axes = numpy.array(axes)
        if len(set(axes % array.ndim)) != len(axes):
            with pytest.raises(ValueError, match='axes must be unique'):
                filter_func(array, *extra_args, size, axes=axes)
            return
        output = filter_func(array, *extra_args, size, axes=axes)
        all_sizes = (size if ax in axes % array.ndim else size0 for ax in range(array.ndim))
        expected = filter_func(array, *extra_args, all_sizes)
        assert_allclose(output, expected)
    kwargs_gauss = dict(radius=[4, 2, 3], order=[0, 1, 2], mode=['reflect', 'nearest', 'constant'])
    kwargs_other = dict(origin=(-1, 0, 1), mode=['reflect', 'nearest', 'constant'])
    kwargs_rank = dict(origin=(-1, 0, 1))

    @pytest.mark.parametrize('filter_func, size0, size, kwargs', [(ndimage.gaussian_filter, 0, 1.0, kwargs_gauss), (ndimage.uniform_filter, 1, 3, kwargs_other), (ndimage.maximum_filter, 1, 3, kwargs_other), (ndimage.minimum_filter, 1, 3, kwargs_other), (ndimage.median_filter, 1, 3, kwargs_rank), (ndimage.rank_filter, 1, 3, kwargs_rank), (ndimage.percentile_filter, 1, 3, kwargs_rank)])
    @pytest.mark.parametrize('axes', itertools.combinations(range(-3, 3), 2))
    def test_filter_axes_kwargs(self, filter_func, size0, size, kwargs, axes):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
        kwargs = {key: numpy.array(val) for key, val in kwargs.items()}
        axes = numpy.array(axes)
        n_axes = axes.size
        if filter_func == ndimage.rank_filter:
            args = (2,)
        elif filter_func == ndimage.percentile_filter:
            args = (30,)
        else:
            args = ()
        reduced_kwargs = {key: val[axes] for key, val in kwargs.items()}
        if len(set(axes % array.ndim)) != len(axes):
            with pytest.raises(ValueError, match='axes must be unique'):
                filter_func(array, *args, [size] * n_axes, axes=axes, **reduced_kwargs)
            return
        output = filter_func(array, *args, [size] * n_axes, axes=axes, **reduced_kwargs)
        size_3d = numpy.full(array.ndim, fill_value=size0)
        size_3d[axes] = size
        if 'origin' in kwargs:
            origin = numpy.array([0, 0, 0])
            origin[axes] = reduced_kwargs['origin']
            kwargs['origin'] = origin
        expected = filter_func(array, *args, size_3d, **kwargs)
        assert_allclose(output, expected)

    @pytest.mark.parametrize('filter_func, args', [(ndimage.gaussian_filter, (1.0,)), (ndimage.uniform_filter, (3,)), (ndimage.minimum_filter, (3,)), (ndimage.maximum_filter, (3,)), (ndimage.median_filter, (3,)), (ndimage.rank_filter, (2, 3)), (ndimage.percentile_filter, (30, 3))])
    @pytest.mark.parametrize('axes', [(1.5,), (0, 1, 2, 3), (3,), (-4,)])
    def test_filter_invalid_axes(self, filter_func, args, axes):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
        if any((isinstance(ax, float) for ax in axes)):
            error_class = TypeError
            match = 'cannot be interpreted as an integer'
        else:
            error_class = ValueError
            match = 'out of range'
        with pytest.raises(error_class, match=match):
            filter_func(array, *args, axes=axes)

    @pytest.mark.parametrize('filter_func, kwargs', [(ndimage.minimum_filter, {}), (ndimage.maximum_filter, {}), (ndimage.median_filter, {}), (ndimage.rank_filter, dict(rank=3)), (ndimage.percentile_filter, dict(percentile=30))])
    @pytest.mark.parametrize('axes', [(0,), (1, 2), (0, 1, 2)])
    @pytest.mark.parametrize('separable_footprint', [False, True])
    def test_filter_invalid_footprint_ndim(self, filter_func, kwargs, axes, separable_footprint):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
        footprint = numpy.ones((3,) * (len(axes) + 1))
        if not separable_footprint:
            footprint[(0,) * footprint.ndim] = 0
        if filter_func in [ndimage.minimum_filter, ndimage.maximum_filter] and separable_footprint:
            match = 'sequence argument must have length equal to input rank'
        else:
            match = 'footprint array has incorrect shape'
        with pytest.raises(RuntimeError, match=match):
            filter_func(array, **kwargs, footprint=footprint, axes=axes)

    @pytest.mark.parametrize('n_mismatch', [1, 3])
    @pytest.mark.parametrize('filter_func, kwargs, key, val', _cases_axes_tuple_length_mismatch())
    def test_filter_tuple_length_mismatch(self, n_mismatch, filter_func, kwargs, key, val):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
        kwargs = dict(**kwargs, axes=(0, 1))
        kwargs[key] = (val,) * n_mismatch
        err_msg = 'sequence argument must have length equal to input rank'
        with pytest.raises(RuntimeError, match=err_msg):
            filter_func(array, **kwargs)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        output = ndimage.prewitt(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.prewitt(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt03(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 0)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt04(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.prewitt(array, -1)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel01(sel, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        output = ndimage.sobel(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.sobel(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel03(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 0)
        output = numpy.zeros(array.shape, dtype)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel04(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        t = ndimage.sobel(array, -1)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_laplace01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        output = ndimage.laplace(array)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_laplace02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.laplace(array, output=output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_gaussian_laplace01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = ndimage.gaussian_laplace(array, 1.0)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_gaussian_laplace02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = numpy.zeros(array.shape, dtype)
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_generic_laplace01(self, dtype):

        def derivative2(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = numpy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 2
            return ndimage.gaussian_filter(input, sigma, order, output, mode, cval)
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = numpy.zeros(array.shape, dtype)
        tmp = ndimage.generic_laplace(array, derivative2, extra_arguments=(1.0,), extra_keywords={'b': 2.0})
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_gaussian_gradient_magnitude01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = ndimage.gaussian_gradient_magnitude(array, 1.0)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = numpy.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_gaussian_gradient_magnitude02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = numpy.zeros(array.shape, dtype)
        ndimage.gaussian_gradient_magnitude(array, 1.0, output)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = numpy.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    def test_generic_gradient_magnitude01(self):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], numpy.float64)

        def derivative(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = numpy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 1
            return ndimage.gaussian_filter(input, sigma, order, output, mode, cval)
        tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
        tmp2 = ndimage.generic_gradient_magnitude(array, derivative, extra_arguments=(1.0,), extra_keywords={'b': 2.0})
        assert_array_almost_equal(tmp1, tmp2)

    def test_uniform01(self):
        array = numpy.array([2, 4, 6])
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal([3, 5, 6], output)

    def test_uniform01_complex(self):
        array = numpy.array([2 + 1j, 4 + 2j, 6 + 3j], dtype=numpy.complex128)
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal([3, 5, 6], output.real)
        assert_array_almost_equal([1.5, 2.5, 3], output.imag)

    def test_uniform02(self):
        array = numpy.array([1, 2, 3])
        filter_shape = [0]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform03(self):
        array = numpy.array([1, 2, 3])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform04(self):
        array = numpy.array([2, 4, 6])
        filter_shape = [2]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 5], output)

    def test_uniform05(self):
        array = []
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([], output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_uniform06(self, dtype_array, dtype_output):
        filter_shape = [2, 2]
        array = numpy.array([[4, 8, 12], [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(array, filter_shape, output=dtype_output)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        assert_equal(output.dtype.type, dtype_output)

    @pytest.mark.parametrize('dtype_array', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_uniform06_complex(self, dtype_array, dtype_output):
        filter_shape = [2, 2]
        array = numpy.array([[4, 8 + 5j, 12], [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(array, filter_shape, output=dtype_output)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output.real)
        assert_equal(output.dtype.type, dtype_output)

    def test_minimum_filter01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([3, 2, 2, 1, 1], output)

    def test_minimum_filter04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([2, 2, 1, 1, 1], output)

    def test_minimum_filter05(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], output)

    def test_minimum_filter05_overlap(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        ndimage.minimum_filter(array, filter_shape, output=array)
        assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], array)

    def test_minimum_filter06(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], output)
        output2 = ndimage.minimum_filter(array, footprint=footprint, mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_minimum_filter07(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]], output)
        with assert_raises(RuntimeError):
            ndimage.minimum_filter(array, footprint=footprint, mode=['reflect', 'constant'])

    def test_minimum_filter08(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal([[3, 1, 3, 1, 1], [5, 3, 3, 1, 1], [3, 3, 1, 1, 1]], output)

    def test_minimum_filter09(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint, origin=[-1, 0])
        assert_array_almost_equal([[2, 3, 1, 3, 1], [5, 5, 3, 3, 1], [5, 3, 3, 1, 1]], output)

    def test_maximum_filter01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([1, 2, 3, 4, 5], output)

    def test_maximum_filter02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 4, 5, 5], output)

    def test_maximum_filter03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_maximum_filter04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 5, 5, 5, 4], output)

    def test_maximum_filter05(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([[3, 5, 5, 5, 4], [7, 9, 9, 9, 5], [8, 9, 9, 9, 7]], output)

    def test_maximum_filter06(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal([[3, 5, 5, 5, 4], [7, 9, 9, 9, 5], [8, 9, 9, 9, 7]], output)
        output2 = ndimage.maximum_filter(array, footprint=footprint, mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_maximum_filter07(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal([[3, 5, 5, 5, 4], [7, 7, 9, 9, 5], [7, 9, 8, 9, 7]], output)
        with assert_raises(RuntimeError):
            ndimage.maximum_filter(array, footprint=footprint, mode=['reflect', 'reflect'])

    def test_maximum_filter08(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal([[7, 9, 9, 5, 5], [9, 8, 9, 7, 5], [8, 8, 7, 7, 7]], output)

    def test_maximum_filter09(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint, origin=[-1, 0])
        assert_array_almost_equal([[7, 7, 9, 9, 5], [7, 9, 8, 9, 7], [8, 8, 8, 7, 7]], output)

    @pytest.mark.parametrize('axes', tuple(itertools.combinations(range(-3, 3), 2)))
    @pytest.mark.parametrize('filter_func, kwargs', [(ndimage.minimum_filter, {}), (ndimage.maximum_filter, {}), (ndimage.median_filter, {}), (ndimage.rank_filter, dict(rank=3)), (ndimage.percentile_filter, dict(percentile=60))])
    def test_minmax_nonseparable_axes(self, filter_func, axes, kwargs):
        array = numpy.arange(6 * 8 * 12, dtype=numpy.float32).reshape(6, 8, 12)
        footprint = numpy.tri(5)
        axes = numpy.array(axes)
        if len(set(axes % array.ndim)) != len(axes):
            with pytest.raises(ValueError):
                filter_func(array, footprint=footprint, axes=axes, **kwargs)
            return
        output = filter_func(array, footprint=footprint, axes=axes, **kwargs)
        missing_axis = tuple(set(range(3)) - set(axes % array.ndim))[0]
        footprint_3d = numpy.expand_dims(footprint, missing_axis)
        expected = filter_func(array, footprint=footprint_3d, **kwargs)
        assert_allclose(output, expected)

    def test_rank01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, 2)
        assert_array_almost_equal(array, output)

    def test_rank02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=[3])
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, (3,))
        assert_array_almost_equal(array, output)

    def test_rank03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        output = ndimage.rank_filter(array, 1, size=[2])
        assert_array_almost_equal([3, 3, 5, 5, 4], output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_rank04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        expected = [3, 3, 2, 4, 4]
        output = ndimage.rank_filter(array, 1, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank05(self):
        array = numpy.array([3, 2, 5, 1, 4])
        expected = [3, 3, 2, 4, 4]
        output = ndimage.rank_filter(array, -2, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank06(self):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
        expected = [[2, 2, 1, 1, 1], [3, 3, 2, 1, 1], [5, 5, 3, 3, 1]]
        output = ndimage.rank_filter(array, 1, size=[2, 3])
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 17, size=(2, 3))
        assert_array_almost_equal(expected, output)

    def test_rank06_overlap(self):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
        array_copy = array.copy()
        expected = [[2, 2, 1, 1, 1], [3, 3, 2, 1, 1], [5, 5, 3, 3, 1]]
        ndimage.rank_filter(array, 1, size=[2, 3], output=array)
        assert_array_almost_equal(expected, array)
        ndimage.percentile_filter(array_copy, 17, size=(2, 3), output=array_copy)
        assert_array_almost_equal(expected, array_copy)

    def test_rank07(self):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
        expected = [[3, 5, 5, 5, 4], [5, 5, 7, 5, 4], [6, 8, 8, 7, 5]]
        output = ndimage.rank_filter(array, -2, size=[2, 3])
        assert_array_almost_equal(expected, output)

    def test_rank08(self):
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]])
        expected = [[3, 3, 2, 4, 4], [5, 5, 5, 4, 4], [5, 6, 7, 5, 5]]
        output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.rank_filter(array, 3, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=(2, 3))
        assert_array_almost_equal(expected, output)
        with assert_raises(RuntimeError):
            ndimage.percentile_filter(array, 50.0, size=(2, 3), mode=['reflect', 'constant'])
        with assert_raises(RuntimeError):
            ndimage.rank_filter(array, 3, size=(2, 3), mode=['reflect'] * 2)
        with assert_raises(RuntimeError):
            ndimage.median_filter(array, size=(2, 3), mode=['reflect'] * 2)

    @pytest.mark.parametrize('dtype', types)
    def test_rank09(self, dtype):
        expected = [[3, 3, 2, 4, 4], [3, 5, 2, 5, 1], [5, 5, 8, 3, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 35, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank10(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        expected = [[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]]
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.rank_filter(array, 0, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 0.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank11(self):
        array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
        expected = [[3, 5, 5, 5, 4], [7, 7, 9, 9, 5], [7, 9, 8, 9, 7]]
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.rank_filter(array, -1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 100.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank12(self, dtype):
        expected = [[3, 3, 2, 4, 4], [3, 5, 2, 5, 1], [5, 5, 8, 3, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50.0, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank13(self, dtype):
        expected = [[5, 2, 5, 1, 1], [5, 8, 3, 5, 5], [6, 6, 5, 5, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint, origin=-1)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank14(self, dtype):
        expected = [[3, 5, 2, 5, 1], [5, 5, 8, 3, 5], [5, 6, 6, 5, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint, origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank15(self, dtype):
        expected = [[2, 3, 1, 4, 1], [5, 3, 7, 1, 1], [5, 5, 3, 3, 3]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 0, footprint=footprint, origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter1d01(self, dtype):
        weights = numpy.array([1.1, 2.2, 3.3])

        def _filter_func(input, output, fltr, total):
            fltr = fltr / total
            for ii in range(input.shape[0] - 2):
                output[ii] = input[ii] * fltr[0]
                output[ii] += input[ii + 1] * fltr[1]
                output[ii] += input[ii + 2] * fltr[2]
        a = numpy.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
        r2 = ndimage.generic_filter1d(a, _filter_func, 3, axis=0, origin=-1, extra_arguments=(weights,), extra_keywords={'total': weights.sum()})
        assert_array_almost_equal(r1, r2)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter01(self, dtype):
        filter_ = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        footprint = numpy.array([[1, 0], [0, 1]])
        cf = numpy.array([1.0, 4.0])

        def _filter_func(buffer, weights, total=1.0):
            weights = cf / total
            return (buffer * weights).sum()
        a = numpy.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate(a, filter_ * footprint)
        if dtype in float_types:
            r1 /= 5
        else:
            r1 //= 5
        r2 = ndimage.generic_filter(a, _filter_func, footprint=footprint, extra_arguments=(cf,), extra_keywords={'total': cf.sum()})
        assert_array_almost_equal(r1, r2)
        with assert_raises(RuntimeError):
            r2 = ndimage.generic_filter(a, _filter_func, mode=['reflect', 'reflect'], footprint=footprint, extra_arguments=(cf,), extra_keywords={'total': cf.sum()})

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [1, 1, 2]), ('wrap', [3, 1, 2]), ('reflect', [1, 1, 2]), ('mirror', [2, 1, 2]), ('constant', [0, 1, 2])])
    def test_extend01(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([1, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [1, 1, 1]), ('wrap', [3, 1, 2]), ('reflect', [3, 3, 2]), ('mirror', [1, 2, 3]), ('constant', [0, 0, 0])])
    def test_extend02(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([1, 0, 0, 0, 0, 0, 0, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [2, 3, 3]), ('wrap', [2, 3, 1]), ('reflect', [2, 3, 3]), ('mirror', [2, 3, 2]), ('constant', [2, 3, 0])])
    def test_extend03(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [3, 3, 3]), ('wrap', [2, 3, 1]), ('reflect', [2, 1, 1]), ('mirror', [1, 2, 3]), ('constant', [0, 0, 0])])
    def test_extend04(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]), ('wrap', [[9, 7, 8], [3, 1, 2], [6, 4, 5]]), ('reflect', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]), ('mirror', [[5, 4, 5], [2, 1, 2], [5, 4, 5]]), ('constant', [[0, 0, 0], [0, 1, 2], [0, 4, 5]])])
    def test_extend05(self, mode, expected_value):
        array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = numpy.array([[1, 0], [0, 0]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]), ('wrap', [[5, 6, 4], [8, 9, 7], [2, 3, 1]]), ('reflect', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]), ('mirror', [[5, 6, 5], [8, 9, 8], [5, 6, 5]]), ('constant', [[5, 6, 0], [8, 9, 0], [0, 0, 0]])])
    def test_extend06(self, mode, expected_value):
        array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [3, 3, 3]), ('wrap', [2, 3, 1]), ('reflect', [2, 1, 1]), ('mirror', [1, 2, 3]), ('constant', [0, 0, 0])])
    def test_extend07(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [[3], [3], [3]]), ('wrap', [[2], [3], [1]]), ('reflect', [[2], [1], [1]]), ('mirror', [[1], [2], [3]]), ('constant', [[0], [0], [0]])])
    def test_extend08(self, mode, expected_value):
        array = numpy.array([[1], [2], [3]])
        weights = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [3, 3, 3]), ('wrap', [2, 3, 1]), ('reflect', [2, 1, 1]), ('mirror', [1, 2, 3]), ('constant', [0, 0, 0])])
    def test_extend09(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize('mode, expected_value', [('nearest', [[3], [3], [3]]), ('wrap', [[2], [3], [1]]), ('reflect', [[2], [1], [1]]), ('mirror', [[1], [2], [3]]), ('constant', [[0], [0], [0]])])
    def test_extend10(self, mode, expected_value):
        array = numpy.array([[1], [2], [3]])
        weights = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)
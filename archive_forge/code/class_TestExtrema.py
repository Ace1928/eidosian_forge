import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
class TestExtrema:

    def test_saturated_arithmetic(self):
        """Adding/subtracting a constant and clipping"""
        data = np.array([[250, 251, 5, 5], [100, 200, 253, 252], [4, 10, 1, 3]], dtype=np.uint8)
        img_constant_added = extrema._add_constant_clip(data, 4)
        expected = np.array([[254, 255, 9, 9], [104, 204, 255, 255], [8, 14, 5, 7]], dtype=np.uint8)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 4)
        expected = np.array([[246, 247, 1, 1], [96, 196, 249, 248], [0, 6, 0, 0]], dtype=np.uint8)
        error = diff(img_constant_subtracted, expected)
        assert error < eps
        data = np.array([[32767, 32766], [-32768, -32767]], dtype=np.int16)
        img_constant_added = extrema._add_constant_clip(data, 1)
        expected = np.array([[32767, 32767], [-32767, -32766]], dtype=np.int16)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 1)
        expected = np.array([[32766, 32765], [-32768, -32768]], dtype=np.int16)
        error = diff(img_constant_subtracted, expected)
        assert error < eps

    def test_h_maxima(self):
        """h-maxima for various data types"""
        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13], [13, 15, 40, 40, 18, 18, 18, 60, 60, 15], [14, 16, 40, 40, 19, 19, 19, 60, 60, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [15, 16, 18, 19, 19, 20, 19, 19, 18, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [14, 16, 80, 80, 19, 19, 19, 100, 100, 16], [13, 15, 80, 80, 18, 18, 18, 100, 100, 15], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]], dtype=np.uint8)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
            data = data.astype(dtype)
            out = extrema.h_maxima(data, 40)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_minima(self):
        """h-minima for various data types"""
        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13], [13, 15, 40, 40, 18, 18, 18, 60, 60, 15], [14, 16, 40, 40, 19, 19, 19, 60, 60, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [15, 16, 18, 19, 19, 20, 19, 19, 18, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [14, 16, 80, 80, 19, 19, 19, 100, 100, 16], [13, 15, 80, 80, 18, 18, 18, 100, 100, 15], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]], dtype=np.uint8)
        data = 100 - data
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
            data = data.astype(dtype)
            out = extrema.h_minima(data, 40)
            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_extrema_float(self):
        """specific tests for float type"""
        data = np.array([[0.1, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13], [0.13, 0.15, 0.4, 0.4, 0.18, 0.18, 0.18, 0.6, 0.6, 0.15], [0.14, 0.16, 0.4, 0.4, 0.19, 0.19, 0.19, 0.6, 0.6, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.15, 0.182, 0.18, 0.19, 0.204, 0.2, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.8, 0.8, 0.19, 0.19, 0.19, 1.0, 1.0, 0.16], [0.13, 0.15, 0.8, 0.8, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13]], dtype=np.float32)
        inverted_data = 1.0 - data
        out = extrema.h_maxima(data, 0.003)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        error = diff(expected_result, out)
        assert error < eps
        out = extrema.h_minima(inverted_data, 0.003)
        error = diff(expected_result, out)
        assert error < eps

    def test_h_maxima_float_image(self):
        """specific tests for h-maxima float image type"""
        w = 10
        x, y = np.mgrid[0:w, 0:w]
        data = 20 - 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
        data[2:4, 2:4] = 40
        data[2:4, 7:9] = 60
        data[7:9, 2:4] = 80
        data[7:9, 7:9] = 100
        data = data.astype(np.float32)
        expected_result = np.zeros_like(data)
        expected_result[data > 19.9] = 1.0
        for h in [1e-12, 1e-06, 0.001, 0.01, 0.1, 0.1]:
            out = extrema.h_maxima(data, h)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_maxima_float_h(self):
        """specific tests for h-maxima float h parameter"""
        data = np.array([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 4, 3, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
        h_vals = np.linspace(1.0, 2.0, 100)
        failures = 0
        for h in h_vals:
            if h % 1 != 0:
                msgs = ['possible precision loss converting image']
            else:
                msgs = []
            with expected_warnings(msgs):
                maxima = extrema.h_maxima(data, h)
            if maxima[2, 2] == 0:
                failures += 1
        assert failures == 0

    def test_h_maxima_large_h(self):
        """test that h-maxima works correctly for large h"""
        data = np.array([[10, 10, 10, 10, 10], [10, 13, 13, 13, 10], [10, 13, 14, 13, 10], [10, 13, 13, 13, 10], [10, 10, 10, 10, 10]], dtype=np.uint8)
        maxima = extrema.h_maxima(data, 5)
        assert np.sum(maxima) == 0
        data = np.array([[10, 10, 10, 10, 10], [10, 13, 13, 13, 10], [10, 13, 14, 13, 10], [10, 13, 13, 13, 10], [10, 10, 10, 10, 10]], dtype=np.float32)
        maxima = extrema.h_maxima(data, 5.0)
        assert np.sum(maxima) == 0

    def test_h_minima_float_image(self):
        """specific tests for h-minima float image type"""
        w = 10
        x, y = np.mgrid[0:w, 0:w]
        data = 180 + 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
        data[2:4, 2:4] = 160
        data[2:4, 7:9] = 140
        data[7:9, 2:4] = 120
        data[7:9, 7:9] = 100
        data = data.astype(np.float32)
        expected_result = np.zeros_like(data)
        expected_result[data < 180.1] = 1.0
        for h in [1e-12, 1e-06, 0.001, 0.01, 0.1, 0.1]:
            out = extrema.h_minima(data, h)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_minima_float_h(self):
        """specific tests for h-minima float h parameter"""
        data = np.array([[4, 4, 4, 4, 4], [4, 1, 1, 1, 4], [4, 1, 0, 1, 4], [4, 1, 1, 1, 4], [4, 4, 4, 4, 4]], dtype=np.uint8)
        h_vals = np.linspace(1.0, 2.0, 100)
        failures = 0
        for h in h_vals:
            if h % 1 != 0:
                msgs = ['possible precision loss converting image']
            else:
                msgs = []
            with expected_warnings(msgs):
                minima = extrema.h_minima(data, h)
            if minima[2, 2] == 0:
                failures += 1
        assert failures == 0

    def test_h_minima_large_h(self):
        """test that h-minima works correctly for large h"""
        data = np.array([[14, 14, 14, 14, 14], [14, 11, 11, 11, 14], [14, 11, 10, 11, 14], [14, 11, 11, 11, 14], [14, 14, 14, 14, 14]], dtype=np.uint8)
        maxima = extrema.h_minima(data, 5)
        assert np.sum(maxima) == 0
        data = np.array([[14, 14, 14, 14, 14], [14, 11, 11, 11, 14], [14, 11, 10, 11, 14], [14, 11, 11, 11, 14], [14, 14, 14, 14, 14]], dtype=np.float32)
        maxima = extrema.h_minima(data, 5.0)
        assert np.sum(maxima) == 0
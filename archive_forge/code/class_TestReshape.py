import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
class TestReshape(unittest.TestCase):

    def test_reshape_2d2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(5, 4)
        got = arr.reshape(5, 4)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_2d1d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(5 * 4)
        got = arr.reshape(5 * 4)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_3d3d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(5, 3, 4)
        got = arr.reshape(5, 3, 4)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_3d2d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(3 * 4, 5)
        got = arr.reshape(3 * 4, 5)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_3d1d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(3 * 4 * 5)
        got = arr.reshape(3 * 4 * 5)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer2d2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(-1, 4)
        got = arr.reshape(-1, 4)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer2d1d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(-1)
        got = arr.reshape(-1)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer3d3d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(5, -1, 4)
        got = arr.reshape(5, -1, 4)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer3d2d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(3, -1)
        got = arr.reshape(3, -1)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer3d1d(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        expect = nparr.reshape(-1)
        got = arr.reshape(-1)[0]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)

    def test_reshape_infer_two_unknowns(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        with self.assertRaises(ValueError) as raises:
            arr.reshape(-1, -1, 3)
        self.assertIn('can only specify one unknown dimension', str(raises.exception))

    def test_reshape_infer_invalid_shape(self):
        nparr = np.empty((3, 4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        with self.assertRaises(ValueError) as raises:
            arr.reshape(-1, 7)
        self.assertIn('cannot infer valid shape for unknown dimension', str(raises.exception))
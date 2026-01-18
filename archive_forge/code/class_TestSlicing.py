import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
class TestSlicing(unittest.TestCase):

    def assertSameContig(self, arr, nparr):
        attrs = ('C_CONTIGUOUS', 'F_CONTIGUOUS')
        for attr in attrs:
            if arr.flags[attr] != nparr.flags[attr]:
                if arr.size == 0 and nparr.size == 0:
                    pass
                else:
                    self.fail('contiguous flag mismatch:\ngot=%s\nexpect=%s' % (arr.flags, nparr.flags))

    def test_slice0_1d(self):
        nparr = np.empty(4)
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        self.assertSameContig(arr, nparr)
        xx = (-2, -1, 0, 1, 2)
        for x in xx:
            expect = nparr[x:]
            got = arr[x:]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_slice1_1d(self):
        nparr = np.empty(4)
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, -1, 0, 1, 2)
        for x in xx:
            expect = nparr[:x]
            got = arr[:x]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_slice2_1d(self):
        nparr = np.empty(4)
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, -1, 0, 1, 2)
        for x, y in itertools.product(xx, xx):
            expect = nparr[x:y]
            got = arr[x:y]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_slice0_2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, 0, 1, 2)
        for x in xx:
            expect = nparr[x:]
            got = arr[x:]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)
        for x, y in itertools.product(xx, xx):
            expect = nparr[x:, y:]
            got = arr[x:, y:]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_slice1_2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, 0, 2)
        for x in xx:
            expect = nparr[:x]
            got = arr[:x]
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)
            self.assertSameContig(got, expect)
        for x, y in itertools.product(xx, xx):
            expect = nparr[:x, :y]
            got = arr[:x, :y]
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)
            self.assertSameContig(got, expect)

    def test_slice2_2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, 0, 2)
        for s, t, u, v in itertools.product(xx, xx, xx, xx):
            expect = nparr[s:t, u:v]
            got = arr[s:t, u:v]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)
        for x, y in itertools.product(xx, xx):
            expect = nparr[s:t, u:v]
            got = arr[s:t, u:v]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_strided_1d(self):
        nparr = np.empty(4)
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, -1, 1, 2)
        for x in xx:
            expect = nparr[::x]
            got = arr[::x]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_strided_2d(self):
        nparr = np.empty((4, 5))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, -1, 1, 2)
        for a, b in itertools.product(xx, xx):
            expect = nparr[::a, ::b]
            got = arr[::a, ::b]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_strided_3d(self):
        nparr = np.empty((4, 5, 6))
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        xx = (-2, -1, 1, 2)
        for a, b, c in itertools.product(xx, xx, xx):
            expect = nparr[::a, ::b, ::c]
            got = arr[::a, ::b, ::c]
            self.assertSameContig(got, expect)
            self.assertEqual(got.shape, expect.shape)
            self.assertEqual(got.strides, expect.strides)

    def test_issue_2766(self):
        z = np.empty((1, 2, 3))
        z = np.transpose(z, axes=(2, 0, 1))
        arr = Array.from_desc(0, z.shape, z.strides, z.itemsize)
        self.assertEqual(z.flags['C_CONTIGUOUS'], arr.flags['C_CONTIGUOUS'])
        self.assertEqual(z.flags['F_CONTIGUOUS'], arr.flags['F_CONTIGUOUS'])
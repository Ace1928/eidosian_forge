import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestStackArrays:

    def setup_method(self):
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.0), ('B', 2.0)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_solo(self):
        _, x, _, _ = self.data
        test = stack_arrays((x,))
        assert_equal(test, x)
        assert_(test is x)
        test = stack_arrays(x)
        assert_equal(test, x)
        assert_(test is x)

    def test_unnamed_fields(self):
        _, x, y, _ = self.data
        test = stack_arrays((x, x), usemask=False)
        control = np.array([1, 2, 1, 2])
        assert_equal(test, control)
        test = stack_arrays((x, y), usemask=False)
        control = np.array([1, 2, 10, 20, 30])
        assert_equal(test, control)
        test = stack_arrays((y, x), usemask=False)
        control = np.array([10, 20, 30, 1, 2])
        assert_equal(test, control)

    def test_unnamed_and_named_fields(self):
        _, x, _, z = self.data
        test = stack_arrays((x, z))
        control = ma.array([(1, -1, -1), (2, -1, -1), (-1, 'A', 1), (-1, 'B', 2)], mask=[(0, 1, 1), (0, 1, 1), (1, 0, 0), (1, 0, 0)], dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        test = stack_arrays((z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1), (-1, -1, 1), (-1, -1, 2)], mask=[(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        test = stack_arrays((z, z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1), ('A', 1, -1), ('B', 2, -1), (-1, -1, 1), (-1, -1, 2)], mask=[(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)

    def test_matching_named_fields(self):
        _, x, _, z = self.data
        zz = np.array([('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)])
        test = stack_arrays((z, zz))
        control = ma.array([('A', 1, -1), ('B', 2, -1), ('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)], mask=[(0, 0, 1), (0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        test = stack_arrays((z, zz, x))
        ndtype = [('A', '|S3'), ('B', float), ('C', float), ('f3', int)]
        control = ma.array([('A', 1, -1, -1), ('B', 2, -1, -1), ('a', 10.0, 100.0, -1), ('b', 20.0, 200.0, -1), ('c', 30.0, 300.0, -1), (-1, -1, -1, 1), (-1, -1, -1, 2)], dtype=ndtype, mask=[(0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (1, 1, 1, 0), (1, 1, 1, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_defaults(self):
        _, _, _, z = self.data
        zz = np.array([('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)])
        defaults = {'A': '???', 'B': -999.0, 'C': -9999.0, 'D': -99999.0}
        test = stack_arrays((z, zz), defaults=defaults)
        control = ma.array([('A', 1, -9999.0), ('B', 2, -9999.0), ('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)], mask=[(0, 0, 1), (0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_autoconversion(self):
        adtype = [('A', int), ('B', bool), ('C', float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [('A', int), ('B', float), ('C', float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)], dtype=bdtype)
        test = stack_arrays((a, b), autoconvert=True)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        with assert_raises(TypeError):
            stack_arrays((a, b), autoconvert=False)

    def test_checktitles(self):
        adtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        test = stack_arrays((a, b))
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)], dtype=bdtype)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_subdtype(self):
        z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float, (1,))])
        zz = np.array([('a', [10.0], 100.0), ('b', [20.0], 200.0), ('c', [30.0], 300.0)], dtype=[('A', '|S3'), ('B', float, (1,)), ('C', float)])
        res = stack_arrays((z, zz))
        expected = ma.array(data=[(b'A', [1.0], 0), (b'B', [2.0], 0), (b'a', [10.0], 100.0), (b'b', [20.0], 200.0), (b'c', [30.0], 300.0)], mask=[(False, [False], True), (False, [False], True), (False, [False], False), (False, [False], False), (False, [False], False)], dtype=zz.dtype)
        assert_equal(res.dtype, expected.dtype)
        assert_equal(res, expected)
        assert_equal(res.mask, expected.mask)
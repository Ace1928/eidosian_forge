import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
class Vector2TypeTest(unittest.TestCase):

    def setUp(self):
        self.zeroVec = Vector2()
        self.e1 = Vector2(1, 0)
        self.e2 = Vector2(0, 1)
        self.t1 = (1.2, 3.4)
        self.l1 = list(self.t1)
        self.v1 = Vector2(self.t1)
        self.t2 = (5.6, 7.8)
        self.l2 = list(self.t2)
        self.v2 = Vector2(self.t2)
        self.s1 = 5.6
        self.s2 = 7.8

    def testConstructionDefault(self):
        v = Vector2()
        self.assertEqual(v.x, 0.0)
        self.assertEqual(v.y, 0.0)

    def testConstructionScalar(self):
        v = Vector2(1)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 1.0)

    def testConstructionScalarKeywords(self):
        v = Vector2(x=1)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 1.0)

    def testConstructionKeywords(self):
        v = Vector2(x=1, y=2)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 2.0)

    def testConstructionXY(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionTuple(self):
        v = Vector2((1.2, 3.4))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionList(self):
        v = Vector2([1.2, 3.4])
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionVector2(self):
        v = Vector2(Vector2(1.2, 3.4))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testAttributeAccess(self):
        tmp = self.v1.x
        self.assertEqual(tmp, self.v1.x)
        self.assertEqual(tmp, self.v1[0])
        tmp = self.v1.y
        self.assertEqual(tmp, self.v1.y)
        self.assertEqual(tmp, self.v1[1])
        self.v1.x = 3.141
        self.assertEqual(self.v1.x, 3.141)
        self.v1.y = 3.141
        self.assertEqual(self.v1.y, 3.141)

        def assign_nonfloat():
            v = Vector2()
            v.x = 'spam'
        self.assertRaises(TypeError, assign_nonfloat)

    def test___round___basic(self):
        self.assertEqual(round(pygame.Vector2(0.0, 0.0)), pygame.Vector2(0.0, 0.0))
        self.assertEqual(type(round(pygame.Vector2(0.0, 0.0))), pygame.Vector2)
        self.assertEqual(round(pygame.Vector2(1.0, 1.0)), round(pygame.Vector2(1.0, 1.0)))
        self.assertEqual(round(pygame.Vector2(10.0, 10.0)), round(pygame.Vector2(10.0, 10.0)))
        self.assertEqual(round(pygame.Vector2(1000000000.0, 1000000000.0)), pygame.Vector2(1000000000.0, 1000000000.0))
        self.assertEqual(round(pygame.Vector2(1e+20, 1e+20)), pygame.Vector2(1e+20, 1e+20))
        self.assertEqual(round(pygame.Vector2(-1.0, -1.0)), pygame.Vector2(-1.0, -1.0))
        self.assertEqual(round(pygame.Vector2(-10.0, -10.0)), pygame.Vector2(-10.0, -10.0))
        self.assertEqual(round(pygame.Vector2(-1000000000.0, -1000000000.0)), pygame.Vector2(-1000000000.0, -1000000000.0))
        self.assertEqual(round(pygame.Vector2(-1e+20, -1e+20)), pygame.Vector2(-1e+20, -1e+20))
        self.assertEqual(round(pygame.Vector2(0.1, 0.1)), pygame.Vector2(0.0, 0.0))
        self.assertEqual(round(pygame.Vector2(1.1, 1.1)), pygame.Vector2(1.0, 1.0))
        self.assertEqual(round(pygame.Vector2(10.1, 10.1)), pygame.Vector2(10.0, 10.0))
        self.assertEqual(round(pygame.Vector2(1000000000.1, 1000000000.1)), pygame.Vector2(1000000000.0, 1000000000.0))
        self.assertEqual(round(pygame.Vector2(-1.1, -1.1)), pygame.Vector2(-1.0, -1.0))
        self.assertEqual(round(pygame.Vector2(-10.1, -10.1)), pygame.Vector2(-10.0, -10.0))
        self.assertEqual(round(pygame.Vector2(-1000000000.1, -1000000000.1)), pygame.Vector2(-1000000000.0, -1000000000.0))
        self.assertEqual(round(pygame.Vector2(0.9, 0.9)), pygame.Vector2(1.0, 1.0))
        self.assertEqual(round(pygame.Vector2(9.9, 9.9)), pygame.Vector2(10.0, 10.0))
        self.assertEqual(round(pygame.Vector2(999999999.9, 999999999.9)), pygame.Vector2(1000000000.0, 1000000000.0))
        self.assertEqual(round(pygame.Vector2(-0.9, -0.9)), pygame.Vector2(-1.0, -1.0))
        self.assertEqual(round(pygame.Vector2(-9.9, -9.9)), pygame.Vector2(-10.0, -10.0))
        self.assertEqual(round(pygame.Vector2(-999999999.9, -999999999.9)), pygame.Vector2(-1000000000.0, -1000000000.0))
        self.assertEqual(round(pygame.Vector2(-8.0, -8.0), -1), pygame.Vector2(-10.0, -10.0))
        self.assertEqual(type(round(pygame.Vector2(-8.0, -8.0), -1)), pygame.Vector2)
        self.assertEqual(type(round(pygame.Vector2(-8.0, -8.0), 0)), pygame.Vector2)
        self.assertEqual(type(round(pygame.Vector2(-8.0, -8.0), 1)), pygame.Vector2)
        self.assertEqual(round(pygame.Vector2(5.5, 5.5)), pygame.Vector2(6, 6))
        self.assertEqual(round(pygame.Vector2(5.4, 5.4)), pygame.Vector2(5.0, 5.0))
        self.assertEqual(round(pygame.Vector2(5.6, 5.6)), pygame.Vector2(6.0, 6.0))
        self.assertEqual(round(pygame.Vector2(-5.5, -5.5)), pygame.Vector2(-6, -6))
        self.assertEqual(round(pygame.Vector2(-5.4, -5.4)), pygame.Vector2(-5, -5))
        self.assertEqual(round(pygame.Vector2(-5.6, -5.6)), pygame.Vector2(-6, -6))
        self.assertRaises(TypeError, round, pygame.Vector2(1.0, 1.0), 1.5)
        self.assertRaises(TypeError, round, pygame.Vector2(1.0, 1.0), 'a')

    def testCopy(self):
        v_copy0 = Vector2(2004.0, 2022.0)
        v_copy1 = v_copy0.copy()
        self.assertEqual(v_copy0.x, v_copy1.x)
        self.assertEqual(v_copy0.y, v_copy1.y)

    def test_move_towards_basic(self):
        expected = Vector2(8.08, 2006.87)
        origin = Vector2(7.22, 2004.0)
        target = Vector2(12.3, 2021.0)
        change_ip = Vector2(7.22, 2004.0)
        change = origin.move_towards(target, 3)
        change_ip.move_towards_ip(target, 3)
        self.assertEqual(round(change.x, 2), expected.x)
        self.assertEqual(round(change.y, 2), expected.y)
        self.assertEqual(round(change_ip.x, 2), expected.x)
        self.assertEqual(round(change_ip.y, 2), expected.y)

    def test_move_towards_max_distance(self):
        expected = Vector2(12.3, 2021)
        origin = Vector2(7.22, 2004.0)
        target = Vector2(12.3, 2021.0)
        change_ip = Vector2(7.22, 2004.0)
        change = origin.move_towards(target, 25)
        change_ip.move_towards_ip(target, 25)
        self.assertEqual(round(change.x, 2), expected.x)
        self.assertEqual(round(change.y, 2), expected.y)
        self.assertEqual(round(change_ip.x, 2), expected.x)
        self.assertEqual(round(change_ip.y, 2), expected.y)

    def test_move_nowhere(self):
        expected = Vector2(7.22, 2004.0)
        origin = Vector2(7.22, 2004.0)
        target = Vector2(12.3, 2021.0)
        change_ip = Vector2(7.22, 2004.0)
        change = origin.move_towards(target, 0)
        change_ip.move_towards_ip(target, 0)
        self.assertEqual(round(change.x, 2), expected.x)
        self.assertEqual(round(change.y, 2), expected.y)
        self.assertEqual(round(change_ip.x, 2), expected.x)
        self.assertEqual(round(change_ip.y, 2), expected.y)

    def test_move_away(self):
        expected = Vector2(6.36, 2001.13)
        origin = Vector2(7.22, 2004.0)
        target = Vector2(12.3, 2021.0)
        change_ip = Vector2(7.22, 2004.0)
        change = origin.move_towards(target, -3)
        change_ip.move_towards_ip(target, -3)
        self.assertEqual(round(change.x, 2), expected.x)
        self.assertEqual(round(change.y, 2), expected.y)
        self.assertEqual(round(change_ip.x, 2), expected.x)
        self.assertEqual(round(change_ip.y, 2), expected.y)

    def test_move_towards_self(self):
        vec = Vector2(6.36, 2001.13)
        vec2 = vec.copy()
        for dist in (-3.54, -1, 0, 0.234, 12):
            self.assertEqual(vec.move_towards(vec2, dist), vec)
            vec2.move_towards_ip(vec, dist)
            self.assertEqual(vec, vec2)

    def test_move_towards_errors(self):

        def overpopulate():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards(target, 3, 2)

        def overpopulate_ip():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards_ip(target, 3, 2)

        def invalid_types1():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards(target, 'novial')

        def invalid_types_ip1():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards_ip(target, 'is')

        def invalid_types2():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards('kinda', 3)

        def invalid_types_ip2():
            origin = Vector2(7.22, 2004.0)
            target = Vector2(12.3, 2021.0)
            origin.move_towards_ip('cool', 3)
        self.assertRaises(TypeError, overpopulate)
        self.assertRaises(TypeError, overpopulate_ip)
        self.assertRaises(TypeError, invalid_types1)
        self.assertRaises(TypeError, invalid_types_ip1)
        self.assertRaises(TypeError, invalid_types2)
        self.assertRaises(TypeError, invalid_types_ip2)

    def testSequence(self):
        v = Vector2(1.2, 3.4)
        Vector2()[:]
        self.assertEqual(len(v), 2)
        self.assertEqual(v[0], 1.2)
        self.assertEqual(v[1], 3.4)
        self.assertRaises(IndexError, lambda: v[2])
        self.assertEqual(v[-1], 3.4)
        self.assertEqual(v[-2], 1.2)
        self.assertRaises(IndexError, lambda: v[-3])
        self.assertEqual(v[:], [1.2, 3.4])
        self.assertEqual(v[1:], [3.4])
        self.assertEqual(v[:1], [1.2])
        self.assertEqual(list(v), [1.2, 3.4])
        self.assertEqual(tuple(v), (1.2, 3.4))
        v[0] = 5.6
        v[1] = 7.8
        self.assertEqual(v.x, 5.6)
        self.assertEqual(v.y, 7.8)
        v[:] = [9.1, 11.12]
        self.assertEqual(v.x, 9.1)
        self.assertEqual(v.y, 11.12)

        def overpopulate():
            v = Vector2()
            v[:] = [1, 2, 3]
        self.assertRaises(ValueError, overpopulate)

        def underpopulate():
            v = Vector2()
            v[:] = [1]
        self.assertRaises(ValueError, underpopulate)

        def assign_nonfloat():
            v = Vector2()
            v[0] = 'spam'
        self.assertRaises(TypeError, assign_nonfloat)

    def testExtendedSlicing(self):

        def delSlice(vec, start=None, stop=None, step=None):
            if start is not None and stop is not None and (step is not None):
                del vec[start:stop:step]
            elif start is not None and stop is None and (step is not None):
                del vec[start::step]
            elif start is None and stop is None and (step is not None):
                del vec[::step]
        v = Vector2(self.v1)
        self.assertRaises(TypeError, delSlice, v, None, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, 2, 1)
        v = Vector2(self.v1)
        v[::2] = [-1]
        self.assertEqual(v, [-1, self.v1.y])
        v = Vector2(self.v1)
        v[::-2] = [10]
        self.assertEqual(v, [self.v1.x, 10])
        v = Vector2(self.v1)
        v[::-1] = v
        self.assertEqual(v, [self.v1.y, self.v1.x])
        a = Vector2(self.v1)
        b = Vector2(self.v1)
        c = Vector2(self.v1)
        a[1:2] = [2.2]
        b[slice(1, 2)] = [2.2]
        c[1:2] = (2.2,)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(type(a), type(self.v1))
        self.assertEqual(type(b), type(self.v1))
        self.assertEqual(type(c), type(self.v1))

    def test_contains(self):
        c = Vector2(0, 1)
        self.assertTrue(c.__contains__(0))
        self.assertTrue(0 in c)
        self.assertTrue(1 in c)
        self.assertTrue(2 not in c)
        self.assertFalse(c.__contains__(2))
        self.assertRaises(TypeError, lambda: 'string' in c)
        self.assertRaises(TypeError, lambda: 3 + 4j in c)

    def testAdd(self):
        v3 = self.v1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.v2.x)
        self.assertEqual(v3.y, self.v1.y + self.v2.y)
        v3 = self.v1 + self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.t2[0])
        self.assertEqual(v3.y, self.v1.y + self.t2[1])
        v3 = self.v1 + self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.l2[0])
        self.assertEqual(v3.y, self.v1.y + self.l2[1])
        v3 = self.t1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] + self.v2.x)
        self.assertEqual(v3.y, self.t1[1] + self.v2.y)
        v3 = self.l1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] + self.v2.x)
        self.assertEqual(v3.y, self.l1[1] + self.v2.y)

    def testSub(self):
        v3 = self.v1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.v2.x)
        self.assertEqual(v3.y, self.v1.y - self.v2.y)
        v3 = self.v1 - self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.t2[0])
        self.assertEqual(v3.y, self.v1.y - self.t2[1])
        v3 = self.v1 - self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.l2[0])
        self.assertEqual(v3.y, self.v1.y - self.l2[1])
        v3 = self.t1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] - self.v2.x)
        self.assertEqual(v3.y, self.t1[1] - self.v2.y)
        v3 = self.l1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] - self.v2.x)
        self.assertEqual(v3.y, self.l1[1] - self.v2.y)

    def testScalarMultiplication(self):
        v = self.s1 * self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.s1 * self.v1.x)
        self.assertEqual(v.y, self.s1 * self.v1.y)
        v = self.v1 * self.s2
        self.assertEqual(v.x, self.v1.x * self.s2)
        self.assertEqual(v.y, self.v1.y * self.s2)

    def testScalarDivision(self):
        v = self.v1 / self.s1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertAlmostEqual(v.x, self.v1.x / self.s1)
        self.assertAlmostEqual(v.y, self.v1.y / self.s1)
        v = self.v1 // self.s2
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x // self.s2)
        self.assertEqual(v.y, self.v1.y // self.s2)

    def testBool(self):
        self.assertEqual(bool(self.zeroVec), False)
        self.assertEqual(bool(self.v1), True)
        self.assertTrue(not self.zeroVec)
        self.assertTrue(self.v1)

    def testUnary(self):
        v = +self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x)
        self.assertEqual(v.y, self.v1.y)
        self.assertNotEqual(id(v), id(self.v1))
        v = -self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, -self.v1.x)
        self.assertEqual(v.y, -self.v1.y)
        self.assertNotEqual(id(v), id(self.v1))

    def testCompare(self):
        int_vec = Vector2(3, -2)
        flt_vec = Vector2(3.0, -2.0)
        zero_vec = Vector2(0, 0)
        self.assertEqual(int_vec == flt_vec, True)
        self.assertEqual(int_vec != flt_vec, False)
        self.assertEqual(int_vec != zero_vec, True)
        self.assertEqual(flt_vec == zero_vec, False)
        self.assertEqual(int_vec == (3, -2), True)
        self.assertEqual(int_vec != (3, -2), False)
        self.assertEqual(int_vec != [0, 0], True)
        self.assertEqual(int_vec == [0, 0], False)
        self.assertEqual(int_vec != 5, True)
        self.assertEqual(int_vec == 5, False)
        self.assertEqual(int_vec != [3, -2, 0], True)
        self.assertEqual(int_vec == [3, -2, 0], False)

    def testStr(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(str(v), '[1.2, 3.4]')

    def testRepr(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(v.__repr__(), '<Vector2(1.2, 3.4)>')
        self.assertEqual(v, Vector2(v.__repr__()))

    def testIter(self):
        it = self.v1.__iter__()
        next_ = it.__next__
        self.assertEqual(next_(), self.v1[0])
        self.assertEqual(next_(), self.v1[1])
        self.assertRaises(StopIteration, lambda: next_())
        it1 = self.v1.__iter__()
        it2 = self.v1.__iter__()
        self.assertNotEqual(id(it1), id(it2))
        self.assertEqual(id(it1), id(it1.__iter__()))
        self.assertEqual(list(it1), list(it2))
        self.assertEqual(list(self.v1.__iter__()), self.l1)
        idx = 0
        for val in self.v1:
            self.assertEqual(val, self.v1[idx])
            idx += 1

    def test_rotate(self):
        v1 = Vector2(1, 0)
        v2 = v1.rotate(90)
        v3 = v1.rotate(90 + 360)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v2.x, 0)
        self.assertEqual(v2.y, 1)
        self.assertEqual(v3.x, v2.x)
        self.assertEqual(v3.y, v2.y)
        v1 = Vector2(-1, -1)
        v2 = v1.rotate(-90)
        self.assertEqual(v2.x, -1)
        self.assertEqual(v2.y, 1)
        v2 = v1.rotate(360)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        v2 = v1.rotate(0)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(Vector2(0, 1).rotate(359.99999999), Vector2(0, 1))

    def test_rotate_rad(self):
        tests = (((1, 0), math.pi), ((1, 0), math.pi / 2), ((1, 0), -math.pi / 2), ((1, 0), math.pi / 4))
        for initialVec, radians in tests:
            self.assertEqual(Vector2(initialVec).rotate_rad(radians), (math.cos(radians), math.sin(radians)))

    def test_rotate_ip(self):
        v = Vector2(1, 0)
        self.assertEqual(v.rotate_ip(90), None)
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 1)
        v = Vector2(-1, -1)
        v.rotate_ip(-90)
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, 1)

    def test_rotate_rad_ip(self):
        tests = (((1, 0), math.pi), ((1, 0), math.pi / 2), ((1, 0), -math.pi / 2), ((1, 0), math.pi / 4))
        for initialVec, radians in tests:
            vec = Vector2(initialVec)
            vec.rotate_rad_ip(radians)
            self.assertEqual(vec, (math.cos(radians), math.sin(radians)))

    def test_normalize(self):
        v = self.v1.normalize()
        self.assertAlmostEqual(v.x * v.x + v.y * v.y, 1.0)
        self.assertEqual(self.v1.x, self.l1[0])
        self.assertEqual(self.v1.y, self.l1[1])
        self.assertAlmostEqual(self.v1.x * v.y - self.v1.y * v.x, 0.0)
        self.assertRaises(ValueError, lambda: self.zeroVec.normalize())

    def test_normalize_ip(self):
        v = +self.v1
        self.assertNotEqual(v.x * v.x + v.y * v.y, 1.0)
        self.assertEqual(v.normalize_ip(), None)
        self.assertAlmostEqual(v.x * v.x + v.y * v.y, 1.0)
        self.assertAlmostEqual(self.v1.x * v.y - self.v1.y * v.x, 0.0)
        self.assertRaises(ValueError, lambda: self.zeroVec.normalize_ip())

    def test_is_normalized(self):
        self.assertEqual(self.v1.is_normalized(), False)
        v = self.v1.normalize()
        self.assertEqual(v.is_normalized(), True)
        self.assertEqual(self.e2.is_normalized(), True)
        self.assertEqual(self.zeroVec.is_normalized(), False)

    def test_cross(self):
        self.assertEqual(self.v1.cross(self.v2), self.v1.x * self.v2.y - self.v1.y * self.v2.x)
        self.assertEqual(self.v1.cross(self.l2), self.v1.x * self.l2[1] - self.v1.y * self.l2[0])
        self.assertEqual(self.v1.cross(self.t2), self.v1.x * self.t2[1] - self.v1.y * self.t2[0])
        self.assertEqual(self.v1.cross(self.v2), -self.v2.cross(self.v1))
        self.assertEqual(self.v1.cross(self.v1), 0)

    def test_dot(self):
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v1.x * self.v2.x + self.v1.y * self.v2.y)
        self.assertAlmostEqual(self.v1.dot(self.l2), self.v1.x * self.l2[0] + self.v1.y * self.l2[1])
        self.assertAlmostEqual(self.v1.dot(self.t2), self.v1.x * self.t2[0] + self.v1.y * self.t2[1])
        self.assertEqual(self.v1.dot(self.v2), self.v2.dot(self.v1))
        self.assertEqual(self.v1.dot(self.v2), self.v1 * self.v2)

    def test_angle_to(self):
        self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2)).normalize(), self.v2.normalize())
        self.assertEqual(Vector2(1, 1).angle_to((-1, 1)), 90)
        self.assertEqual(Vector2(1, 0).angle_to((0, -1)), -90)
        self.assertEqual(Vector2(1, 0).angle_to((-1, 1)), 135)
        self.assertEqual(abs(Vector2(1, 0).angle_to((-1, 0))), 180)

    def test_scale_to_length(self):
        v = Vector2(1, 1)
        v.scale_to_length(2.5)
        self.assertEqual(v, Vector2(2.5, 2.5) / math.sqrt(2))
        self.assertRaises(ValueError, lambda: self.zeroVec.scale_to_length(1))
        self.assertEqual(v.scale_to_length(0), None)
        self.assertEqual(v, self.zeroVec)

    def test_length(self):
        self.assertEqual(Vector2(3, 4).length(), 5)
        self.assertEqual(Vector2(-3, 4).length(), 5)
        self.assertEqual(self.zeroVec.length(), 0)

    def test_length_squared(self):
        self.assertEqual(Vector2(3, 4).length_squared(), 25)
        self.assertEqual(Vector2(-3, 4).length_squared(), 25)
        self.assertEqual(self.zeroVec.length_squared(), 0)

    def test_reflect(self):
        v = Vector2(1, -1)
        n = Vector2(0, 1)
        self.assertEqual(v.reflect(n), Vector2(1, 1))
        self.assertEqual(v.reflect(3 * n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ValueError, lambda: v.reflect(self.zeroVec))

    def test_reflect_ip(self):
        v1 = Vector2(1, -1)
        v2 = Vector2(v1)
        n = Vector2(0, 1)
        self.assertEqual(v2.reflect_ip(n), None)
        self.assertEqual(v2, Vector2(1, 1))
        v2 = Vector2(v1)
        v2.reflect_ip(3 * n)
        self.assertEqual(v2, v1.reflect(n))
        v2 = Vector2(v1)
        v2.reflect_ip(-v1)
        self.assertEqual(v2, -v1)
        self.assertRaises(ValueError, lambda: v2.reflect_ip(Vector2()))

    def test_distance_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_to(self.e2), math.sqrt(2))
        self.assertEqual(self.e1.distance_to((0, 1)), math.sqrt(2))
        self.assertEqual(self.e1.distance_to([0, 1]), math.sqrt(2))
        self.assertAlmostEqual(self.v1.distance_to(self.v2), math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertAlmostEqual(self.v1.distance_to(self.t2), math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertAlmostEqual(self.v1.distance_to(self.l2), math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertEqual(self.v1.distance_to(self.v1), 0)
        self.assertEqual(self.v1.distance_to(self.t1), 0)
        self.assertEqual(self.v1.distance_to(self.l1), 0)
        self.assertEqual(self.v1.distance_to(self.t2), self.v2.distance_to(self.t1))
        self.assertEqual(self.v1.distance_to(self.l2), self.v2.distance_to(self.l1))
        self.assertEqual(self.v1.distance_to(self.v2), self.v2.distance_to(self.v1))

    def test_distance_squared_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_squared_to(self.e2), 2)
        self.assertEqual(self.e1.distance_squared_to((0, 1)), 2)
        self.assertEqual(self.e1.distance_squared_to([0, 1]), 2)
        self.assertAlmostEqual(self.v1.distance_squared_to(self.v2), diff.x * diff.x + diff.y * diff.y)
        self.assertAlmostEqual(self.v1.distance_squared_to(self.t2), diff.x * diff.x + diff.y * diff.y)
        self.assertAlmostEqual(self.v1.distance_squared_to(self.l2), diff.x * diff.x + diff.y * diff.y)
        self.assertEqual(self.v1.distance_squared_to(self.v1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.t1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.l1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.v2), self.v2.distance_squared_to(self.v1))
        self.assertEqual(self.v1.distance_squared_to(self.t2), self.v2.distance_squared_to(self.t1))
        self.assertEqual(self.v1.distance_squared_to(self.l2), self.v2.distance_squared_to(self.l1))

    def test_update(self):
        v = Vector2(3, 4)
        v.update(0)
        self.assertEqual(v, Vector2((0, 0)))
        v.update(5, 1)
        self.assertEqual(v, Vector2(5, 1))
        v.update((4, 1))
        self.assertNotEqual(v, Vector2((5, 1)))

    def test_swizzle(self):
        self.assertEqual(self.v1.yx, (self.v1.y, self.v1.x))
        self.assertEqual(self.v1.xxyyxy, (self.v1.x, self.v1.x, self.v1.y, self.v1.y, self.v1.x, self.v1.y))
        self.v1.xy = self.t2
        self.assertEqual(self.v1, self.t2)
        self.v1.yx = self.t2
        self.assertEqual(self.v1, (self.t2[1], self.t2[0]))
        self.assertEqual(type(self.v1), Vector2)

        def invalidSwizzleX():
            Vector2().xx = (1, 2)

        def invalidSwizzleY():
            Vector2().yy = (1, 2)
        self.assertRaises(AttributeError, invalidSwizzleX)
        self.assertRaises(AttributeError, invalidSwizzleY)

        def invalidAssignment():
            Vector2().xy = 3
        self.assertRaises(TypeError, invalidAssignment)

        def unicodeAttribute():
            getattr(Vector2(), 'ä')
        self.assertRaises(AttributeError, unicodeAttribute)

    def test_swizzle_return_types(self):
        self.assertEqual(type(self.v1.x), float)
        self.assertEqual(type(self.v1.xy), Vector2)
        self.assertEqual(type(self.v1.xyx), Vector3)
        self.assertEqual(type(self.v1.xyxy), tuple)
        self.assertEqual(type(self.v1.xyxyx), tuple)

    def test_elementwise(self):
        v1 = self.v1
        v2 = self.v2
        s1 = self.s1
        s2 = self.s2
        self.assertEqual(v1.elementwise() + s1, (v1.x + s1, v1.y + s1))
        self.assertEqual(v1.elementwise() - s1, (v1.x - s1, v1.y - s1))
        self.assertEqual(v1.elementwise() * s2, (v1.x * s2, v1.y * s2))
        self.assertEqual(v1.elementwise() / s2, (v1.x / s2, v1.y / s2))
        self.assertEqual(v1.elementwise() // s1, (v1.x // s1, v1.y // s1))
        self.assertEqual(v1.elementwise() ** s1, (v1.x ** s1, v1.y ** s1))
        self.assertEqual(v1.elementwise() % s1, (v1.x % s1, v1.y % s1))
        self.assertEqual(v1.elementwise() > s1, v1.x > s1 and v1.y > s1)
        self.assertEqual(v1.elementwise() < s1, v1.x < s1 and v1.y < s1)
        self.assertEqual(v1.elementwise() == s1, v1.x == s1 and v1.y == s1)
        self.assertEqual(v1.elementwise() != s1, s1 not in [v1.x, v1.y])
        self.assertEqual(v1.elementwise() >= s1, v1.x >= s1 and v1.y >= s1)
        self.assertEqual(v1.elementwise() <= s1, v1.x <= s1 and v1.y <= s1)
        self.assertEqual(v1.elementwise() != s1, s1 not in [v1.x, v1.y])
        self.assertEqual(s1 + v1.elementwise(), (s1 + v1.x, s1 + v1.y))
        self.assertEqual(s1 - v1.elementwise(), (s1 - v1.x, s1 - v1.y))
        self.assertEqual(s1 * v1.elementwise(), (s1 * v1.x, s1 * v1.y))
        self.assertEqual(s1 / v1.elementwise(), (s1 / v1.x, s1 / v1.y))
        self.assertEqual(s1 // v1.elementwise(), (s1 // v1.x, s1 // v1.y))
        self.assertEqual(s1 ** v1.elementwise(), (s1 ** v1.x, s1 ** v1.y))
        self.assertEqual(s1 % v1.elementwise(), (s1 % v1.x, s1 % v1.y))
        self.assertEqual(s1 < v1.elementwise(), s1 < v1.x and s1 < v1.y)
        self.assertEqual(s1 > v1.elementwise(), s1 > v1.x and s1 > v1.y)
        self.assertEqual(s1 == v1.elementwise(), s1 == v1.x and s1 == v1.y)
        self.assertEqual(s1 != v1.elementwise(), s1 not in [v1.x, v1.y])
        self.assertEqual(s1 <= v1.elementwise(), s1 <= v1.x and s1 <= v1.y)
        self.assertEqual(s1 >= v1.elementwise(), s1 >= v1.x and s1 >= v1.y)
        self.assertEqual(s1 != v1.elementwise(), s1 not in [v1.x, v1.y])
        self.assertEqual(type(v1.elementwise() * v2), type(v1))
        self.assertEqual(v1.elementwise() + v2, v1 + v2)
        self.assertEqual(v1.elementwise() - v2, v1 - v2)
        self.assertEqual(v1.elementwise() * v2, (v1.x * v2.x, v1.y * v2.y))
        self.assertEqual(v1.elementwise() / v2, (v1.x / v2.x, v1.y / v2.y))
        self.assertEqual(v1.elementwise() // v2, (v1.x // v2.x, v1.y // v2.y))
        self.assertEqual(v1.elementwise() ** v2, (v1.x ** v2.x, v1.y ** v2.y))
        self.assertEqual(v1.elementwise() % v2, (v1.x % v2.x, v1.y % v2.y))
        self.assertEqual(v1.elementwise() > v2, v1.x > v2.x and v1.y > v2.y)
        self.assertEqual(v1.elementwise() < v2, v1.x < v2.x and v1.y < v2.y)
        self.assertEqual(v1.elementwise() >= v2, v1.x >= v2.x and v1.y >= v2.y)
        self.assertEqual(v1.elementwise() <= v2, v1.x <= v2.x and v1.y <= v2.y)
        self.assertEqual(v1.elementwise() == v2, v1.x == v2.x and v1.y == v2.y)
        self.assertEqual(v1.elementwise() != v2, v1.x != v2.x and v1.y != v2.y)
        self.assertEqual(v2 + v1.elementwise(), v2 + v1)
        self.assertEqual(v2 - v1.elementwise(), v2 - v1)
        self.assertEqual(v2 * v1.elementwise(), (v2.x * v1.x, v2.y * v1.y))
        self.assertEqual(v2 / v1.elementwise(), (v2.x / v1.x, v2.y / v1.y))
        self.assertEqual(v2 // v1.elementwise(), (v2.x // v1.x, v2.y // v1.y))
        self.assertEqual(v2 ** v1.elementwise(), (v2.x ** v1.x, v2.y ** v1.y))
        self.assertEqual(v2 % v1.elementwise(), (v2.x % v1.x, v2.y % v1.y))
        self.assertEqual(v2 < v1.elementwise(), v2.x < v1.x and v2.y < v1.y)
        self.assertEqual(v2 > v1.elementwise(), v2.x > v1.x and v2.y > v1.y)
        self.assertEqual(v2 <= v1.elementwise(), v2.x <= v1.x and v2.y <= v1.y)
        self.assertEqual(v2 >= v1.elementwise(), v2.x >= v1.x and v2.y >= v1.y)
        self.assertEqual(v2 == v1.elementwise(), v2.x == v1.x and v2.y == v1.y)
        self.assertEqual(v2 != v1.elementwise(), v2.x != v1.x and v2.y != v1.y)
        self.assertEqual(v2.elementwise() + v1.elementwise(), v2 + v1)
        self.assertEqual(v2.elementwise() - v1.elementwise(), v2 - v1)
        self.assertEqual(v2.elementwise() * v1.elementwise(), (v2.x * v1.x, v2.y * v1.y))
        self.assertEqual(v2.elementwise() / v1.elementwise(), (v2.x / v1.x, v2.y / v1.y))
        self.assertEqual(v2.elementwise() // v1.elementwise(), (v2.x // v1.x, v2.y // v1.y))
        self.assertEqual(v2.elementwise() ** v1.elementwise(), (v2.x ** v1.x, v2.y ** v1.y))
        self.assertEqual(v2.elementwise() % v1.elementwise(), (v2.x % v1.x, v2.y % v1.y))
        self.assertEqual(v2.elementwise() < v1.elementwise(), v2.x < v1.x and v2.y < v1.y)
        self.assertEqual(v2.elementwise() > v1.elementwise(), v2.x > v1.x and v2.y > v1.y)
        self.assertEqual(v2.elementwise() <= v1.elementwise(), v2.x <= v1.x and v2.y <= v1.y)
        self.assertEqual(v2.elementwise() >= v1.elementwise(), v2.x >= v1.x and v2.y >= v1.y)
        self.assertEqual(v2.elementwise() == v1.elementwise(), v2.x == v1.x and v2.y == v1.y)
        self.assertEqual(v2.elementwise() != v1.elementwise(), v2.x != v1.x and v2.y != v1.y)
        self.assertEqual(abs(v1.elementwise()), (abs(v1.x), abs(v1.y)))
        self.assertEqual(-v1.elementwise(), -v1)
        self.assertEqual(+v1.elementwise(), +v1)
        self.assertEqual(bool(v1.elementwise()), bool(v1))
        self.assertEqual(bool(Vector2().elementwise()), bool(Vector2()))
        self.assertEqual(self.zeroVec.elementwise() ** 0, (1, 1))
        self.assertRaises(ValueError, lambda: pow(Vector2(-1, 0).elementwise(), 1.2))
        self.assertRaises(ZeroDivisionError, lambda: self.zeroVec.elementwise() ** (-1))
        self.assertRaises(ZeroDivisionError, lambda: self.zeroVec.elementwise() ** (-1))
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() / 0)
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() // 0)
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() % 0)
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() / self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() // self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda: Vector2(1, 1).elementwise() % self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda: 2 / self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda: 2 // self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda: 2 % self.zeroVec.elementwise())

    def test_slerp(self):
        self.assertRaises(ValueError, lambda: self.zeroVec.slerp(self.v1, 0.5))
        self.assertRaises(ValueError, lambda: self.v1.slerp(self.zeroVec, 0.5))
        self.assertRaises(ValueError, lambda: self.zeroVec.slerp(self.zeroVec, 0.5))
        v1 = Vector2(1, 0)
        v2 = Vector2(0, 1)
        steps = 10
        angle_step = v1.angle_to(v2) / steps
        for i, u in ((i, v1.slerp(v2, i / float(steps))) for i in range(steps + 1)):
            self.assertAlmostEqual(u.length(), 1)
            self.assertAlmostEqual(v1.angle_to(u), i * angle_step)
        self.assertEqual(u, v2)
        v1 = Vector2(100, 0)
        v2 = Vector2(0, 10)
        radial_factor = v2.length() / v1.length()
        for i, u in ((i, v1.slerp(v2, -i / float(steps))) for i in range(steps + 1)):
            self.assertAlmostEqual(u.length(), (v2.length() - v1.length()) * (float(i) / steps) + v1.length())
        self.assertEqual(u, v2)
        self.assertEqual(v1.slerp(v1, 0.5), v1)
        self.assertEqual(v2.slerp(v2, 0.5), v2)
        self.assertRaises(ValueError, lambda: v1.slerp(-v1, 0.5))

    def test_lerp(self):
        v1 = Vector2(0, 0)
        v2 = Vector2(10, 10)
        self.assertEqual(v1.lerp(v2, 0.5), (5, 5))
        self.assertRaises(ValueError, lambda: v1.lerp(v2, 2.5))
        v1 = Vector2(-10, -5)
        v2 = Vector2(10, 10)
        self.assertEqual(v1.lerp(v2, 0.5), (0, 2.5))

    def test_polar(self):
        v = Vector2()
        v.from_polar(self.v1.as_polar())
        self.assertEqual(self.v1, v)
        self.assertEqual(self.v1, Vector2.from_polar(self.v1.as_polar()))
        self.assertEqual(self.e1.as_polar(), (1, 0))
        self.assertEqual(self.e2.as_polar(), (1, 90))
        self.assertEqual((2 * self.e2).as_polar(), (2, 90))
        self.assertRaises(TypeError, lambda: v.from_polar((None, None)))
        self.assertRaises(TypeError, lambda: v.from_polar('ab'))
        self.assertRaises(TypeError, lambda: v.from_polar((None, 1)))
        self.assertRaises(TypeError, lambda: v.from_polar((1, 2, 3)))
        self.assertRaises(TypeError, lambda: v.from_polar((1,)))
        self.assertRaises(TypeError, lambda: v.from_polar(1, 2))
        self.assertRaises(TypeError, lambda: Vector2.from_polar((None, None)))
        self.assertRaises(TypeError, lambda: Vector2.from_polar('ab'))
        self.assertRaises(TypeError, lambda: Vector2.from_polar((None, 1)))
        self.assertRaises(TypeError, lambda: Vector2.from_polar((1, 2, 3)))
        self.assertRaises(TypeError, lambda: Vector2.from_polar((1,)))
        self.assertRaises(TypeError, lambda: Vector2.from_polar(1, 2))
        v.from_polar((0.5, 90))
        self.assertEqual(v, 0.5 * self.e2)
        self.assertEqual(Vector2.from_polar((0.5, 90)), 0.5 * self.e2)
        self.assertEqual(Vector2.from_polar((0.5, 90)), v)
        v.from_polar((1, 0))
        self.assertEqual(v, self.e1)
        self.assertEqual(Vector2.from_polar((1, 0)), self.e1)
        self.assertEqual(Vector2.from_polar((1, 0)), v)

    def test_subclass_operation(self):

        class Vector(pygame.math.Vector2):
            pass
        vec = Vector()
        vec_a = Vector(2, 0)
        vec_b = Vector(0, 1)
        vec_a + vec_b
        vec_a *= 2

    def test_project_v2_onto_x_axis(self):
        """Project onto x-axis, e.g. get the component pointing in the x-axis direction."""
        v = Vector2(2, 2)
        x_axis = Vector2(10, 0)
        actual = v.project(x_axis)
        self.assertEqual(v.x, actual.x)
        self.assertEqual(0, actual.y)

    def test_project_v2_onto_y_axis(self):
        """Project onto y-axis, e.g. get the component pointing in the y-axis direction."""
        v = Vector2(2, 2)
        y_axis = Vector2(0, 100)
        actual = v.project(y_axis)
        self.assertEqual(0, actual.x)
        self.assertEqual(v.y, actual.y)

    def test_project_v2_onto_other(self):
        """Project onto other vector."""
        v = Vector2(2, 3)
        other = Vector2(3, 5)
        actual = v.project(other)
        expected = v.dot(other) / other.dot(other) * other
        self.assertEqual(expected.x, actual.x)
        self.assertEqual(expected.y, actual.y)

    def test_project_v2_onto_other_as_tuple(self):
        """Project onto other tuple as vector."""
        v = Vector2(2, 3)
        other = Vector2(3, 5)
        actual = v.project(tuple(other))
        expected = v.dot(other) / other.dot(other) * other
        self.assertEqual(expected.x, actual.x)
        self.assertEqual(expected.y, actual.y)

    def test_project_v2_onto_other_as_list(self):
        """Project onto other list as vector."""
        v = Vector2(2, 3)
        other = Vector2(3, 5)
        actual = v.project(list(other))
        expected = v.dot(other) / other.dot(other) * other
        self.assertEqual(expected.x, actual.x)
        self.assertEqual(expected.y, actual.y)

    def test_project_v2_raises_if_other_has_zero_length(self):
        """Check if exception is raise when projected on vector has zero length."""
        v = Vector2(2, 3)
        other = Vector2(0, 0)
        self.assertRaises(ValueError, v.project, other)

    def test_project_v2_raises_if_other_is_not_iterable(self):
        """Check if exception is raise when projected on vector is not iterable."""
        v = Vector2(2, 3)
        other = 10
        self.assertRaises(TypeError, v.project, other)

    def test_collection_abc(self):
        v = Vector2(3, 4)
        self.assertTrue(isinstance(v, Collection))
        self.assertFalse(isinstance(v, Sequence))

    def test_clamp_mag_v2_max(self):
        v1 = Vector2(7, 2)
        v2 = v1.clamp_magnitude(5)
        v3 = v1.clamp_magnitude(0, 5)
        self.assertEqual(v2, v3)
        v1.clamp_magnitude_ip(5)
        self.assertEqual(v1, v2)
        v1.clamp_magnitude_ip(0, 5)
        self.assertEqual(v1, v2)
        expected_v2 = Vector2(4.807619738204116, 1.3736056394868903)
        self.assertEqual(expected_v2, v2)

    def test_clamp_mag_v2_min(self):
        v1 = Vector2(1, 2)
        v2 = v1.clamp_magnitude(3, 5)
        v1.clamp_magnitude_ip(3, 5)
        expected_v2 = Vector2(1.3416407864998738, 2.6832815729997477)
        self.assertEqual(expected_v2, v2)
        self.assertEqual(expected_v2, v1)

    def test_clamp_mag_v2_no_change(self):
        v1 = Vector2(1, 2)
        for args in ((1, 6), (1.12, 3.55), (0.93, 2.83), (7.6,)):
            with self.subTest(args=args):
                v2 = v1.clamp_magnitude(*args)
                v1.clamp_magnitude_ip(*args)
                self.assertEqual(v1, v2)
                self.assertEqual(v1, Vector2(1, 2))

    def test_clamp_mag_v2_edge_cases(self):
        v1 = Vector2(1, 2)
        v2 = v1.clamp_magnitude(6, 6)
        v1.clamp_magnitude_ip(6, 6)
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(v1.length(), 6)
        v2 = v1.clamp_magnitude(0)
        v1.clamp_magnitude_ip(0, 0)
        self.assertEqual(v1, v2)
        self.assertEqual(v1, Vector2())

    def test_clamp_mag_v2_errors(self):
        v1 = Vector2(1, 2)
        for invalid_args in (('foo', 'bar'), (1, 2, 3), (342.234, 'test')):
            with self.subTest(invalid_args=invalid_args):
                self.assertRaises(TypeError, v1.clamp_magnitude, *invalid_args)
                self.assertRaises(TypeError, v1.clamp_magnitude_ip, *invalid_args)
        for invalid_args in ((-1,), (4, 3), (-4, 10), (-4, -2)):
            with self.subTest(invalid_args=invalid_args):
                self.assertRaises(ValueError, v1.clamp_magnitude, *invalid_args)
                self.assertRaises(ValueError, v1.clamp_magnitude_ip, *invalid_args)
        v2 = Vector2()
        self.assertRaises(ValueError, v2.clamp_magnitude, 3)
        self.assertRaises(ValueError, v2.clamp_magnitude_ip, 4)

    def test_subclassing_v2(self):
        """Check if Vector2 is subclassable"""
        v = Vector2(4, 2)

        class TestVector(Vector2):

            def supermariobrosiscool(self):
                return 722
        other = TestVector(4, 1)
        self.assertEqual(other.supermariobrosiscool(), 722)
        self.assertNotEqual(type(v), TestVector)
        self.assertNotEqual(type(v), type(other.copy()))
        self.assertEqual(TestVector, type(other.reflect(v)))
        self.assertEqual(TestVector, type(other.lerp(v, 1)))
        self.assertEqual(TestVector, type(other.slerp(v, 1)))
        self.assertEqual(TestVector, type(other.rotate(5)))
        self.assertEqual(TestVector, type(other.rotate_rad(5)))
        self.assertEqual(TestVector, type(other.project(v)))
        self.assertEqual(TestVector, type(other.move_towards(v, 5)))
        self.assertEqual(TestVector, type(other.clamp_magnitude(5)))
        self.assertEqual(TestVector, type(other.clamp_magnitude(1, 5)))
        self.assertEqual(TestVector, type(other.elementwise() + other))
        other1 = TestVector(4, 2)
        self.assertEqual(type(other + other1), TestVector)
        self.assertEqual(type(other - other1), TestVector)
        self.assertEqual(type(other * 3), TestVector)
        self.assertEqual(type(other / 3), TestVector)
        self.assertEqual(type(other.elementwise() ** 3), TestVector)
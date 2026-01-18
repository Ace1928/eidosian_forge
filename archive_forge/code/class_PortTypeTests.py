import re
import unittest
from oslo_config import types
class PortTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Port()

    def test_port(self):
        self.assertInvalid(-1)
        self.assertInvalid(65536)
        self.assertConvertedValue('80', 80)
        self.assertConvertedValue('65535', 65535)

    def test_repr(self):
        self.assertEqual('Port(min=0, max=65535)', repr(types.Port()))

    def test_repr_with_min(self):
        t = types.Port(min=123)
        self.assertEqual('Port(min=123, max=65535)', repr(t))

    def test_repr_with_max(self):
        t = types.Port(max=456)
        self.assertEqual('Port(min=0, max=456)', repr(t))

    def test_repr_with_min_and_max(self):
        t = types.Port(min=123, max=456)
        self.assertEqual('Port(min=123, max=456)', repr(t))
        t = types.Port(min=0, max=0)
        self.assertEqual('Port(min=0, max=0)', repr(t))

    def test_repr_with_choices(self):
        t = types.Port(choices=[80, 457])
        self.assertEqual('Port(choices=[80, 457])', repr(t))

    def test_repr_with_choices_tuple(self):
        t = types.Port(choices=(80, 457))
        self.assertEqual('Port(choices=[80, 457])', repr(t))

    def _test_with_choices(self, t):
        self.assertRaises(ValueError, t, 1)
        self.assertRaises(ValueError, t, 200)
        self.assertRaises(ValueError, t, -457)
        t(80)
        t(457)

    def test_with_choices_list(self):
        t = types.Port(choices=[80, 457])
        self._test_with_choices(t)

    def test_with_choices_tuple(self):
        t = types.Port(choices=(80, 457))
        self._test_with_choices(t)

    def test_with_choices_dict(self):
        t = types.Port(choices=[(80, 'ab'), (457, 'xy')])
        self._test_with_choices(t)

    def test_invalid_choices(self):
        """Check for choices that are specifically invalid for ports."""
        self.assertRaises(ValueError, types.Port, choices=[-1, 457])
        self.assertRaises(ValueError, types.Port, choices=[1, 2, 3, 65536])

    def test_equal(self):
        self.assertTrue(types.Port() == types.Port())

    def test_equal_with_same_min_and_no_max(self):
        self.assertTrue(types.Port(min=123) == types.Port(min=123))

    def test_equal_with_same_max_and_no_min(self):
        self.assertTrue(types.Port(max=123) == types.Port(max=123))

    def test_equal_with_same_min_and_max(self):
        t1 = types.Port(min=1, max=123)
        t2 = types.Port(min=1, max=123)
        self.assertTrue(t1 == t2)

    def test_equal_with_same_choices(self):
        t1 = types.Port(choices=[80, 457])
        t2 = types.Port(choices=[457, 80])
        t3 = types.Port(choices=(457, 80))
        t4 = types.Port(choices=[(457, 'ab'), (80, 'xy')])
        self.assertTrue(t1 == t2 == t3 == t4)

    def test_not_equal(self):
        self.assertFalse(types.Port(min=123) == types.Port(min=456))
        self.assertFalse(types.Port(choices=[80, 457]) == types.Port(choices=[80, 40]))
        self.assertFalse(types.Port(choices=[80, 457]) == types.Port())

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.Port() == types.Integer())

    def test_choices_with_min_max(self):
        self.assertRaises(ValueError, types.Port, min=100, choices=[50, 60])
        self.assertRaises(ValueError, types.Port, max=10, choices=[50, 60])
        types.Port(min=10, max=100, choices=[50, 60])

    def test_min_greater_max(self):
        self.assertRaises(ValueError, types.Port, min=100, max=50)
        self.assertRaises(ValueError, types.Port, min=-50, max=-100)
        self.assertRaises(ValueError, types.Port, min=0, max=-50)
        self.assertRaises(ValueError, types.Port, min=50, max=0)

    def test_illegal_min(self):
        self.assertRaises(ValueError, types.Port, min=-1, max=50)
        self.assertRaises(ValueError, types.Port, min=-50)

    def test_illegal_max(self):
        self.assertRaises(ValueError, types.Port, min=100, max=65537)
        self.assertRaises(ValueError, types.Port, max=100000)

    def test_with_max_and_min(self):
        t = types.Port(min=123, max=456)
        self.assertRaises(ValueError, t, 122)
        t(123)
        t(300)
        t(456)
        self.assertRaises(ValueError, t, 0)
        self.assertRaises(ValueError, t, 457)

    def test_with_min_zero(self):
        t = types.Port(min=0, max=456)
        self.assertRaises(ValueError, t, -1)
        t(0)
        t(123)
        t(300)
        t(456)
        self.assertRaises(ValueError, t, -201)
        self.assertRaises(ValueError, t, 457)

    def test_with_max_zero(self):
        t = types.Port(max=0)
        self.assertRaises(ValueError, t, 1)
        t(0)
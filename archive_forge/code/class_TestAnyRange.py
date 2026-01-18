import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
class TestAnyRange(unittest.TestCase):

    def test_str(self):
        self.assertEqual(str(AnyRange()), '[*]')

    def test_range_relational(self):
        a = AnyRange()
        b = AnyRange()
        self.assertTrue(a.issubset(b))
        self.assertEqual(a, a)
        self.assertEqual(a, b)
        c = NR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertTrue(c.issubset(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(c, a)

    def test_contains(self):
        a = AnyRange()
        self.assertIn(None, a)
        self.assertIn(0, a)
        self.assertIn('a', a)

    def test_range_difference(self):
        self.assertEqual(AnyRange().range_difference([NR(0, None, 1)]), [AnyRange()])
        self.assertEqual(NR(0, None, 1).range_difference([AnyRange()]), [])
        self.assertEqual(AnyRange().range_difference([AnyRange()]), [])

    def test_range_intersection(self):
        self.assertEqual(AnyRange().range_intersection([NR(0, None, 1)]), [NR(0, None, 1)])
        self.assertEqual(NR(0, None, 1).range_intersection([AnyRange()]), [NR(0, None, 1)])
        self.assertEqual(NR(0, None, -1).range_intersection([AnyRange()]), [NR(0, None, -1)])

    def test_info_methods(self):
        a = AnyRange()
        self.assertFalse(a.isdiscrete())
        self.assertFalse(a.isfinite())

    def test_pickle(self):
        a = AnyRange()
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
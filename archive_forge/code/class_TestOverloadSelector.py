from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
class TestOverloadSelector(unittest.TestCase):

    def test_select_and_sort_1(self):
        os = OverloadSelector()
        os.append(1, (types.Any, types.Boolean))
        os.append(2, (types.Boolean, types.Integer))
        os.append(3, (types.Boolean, types.Any))
        os.append(4, (types.Boolean, types.Boolean))
        compats = os._select_compatible((types.boolean, types.boolean))
        self.assertEqual(len(compats), 3)
        ordered, scoring = os._sort_signatures(compats)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(len(scoring), 3)
        self.assertEqual(ordered[0], (types.Boolean, types.Boolean))
        self.assertEqual(scoring[types.Boolean, types.Boolean], 0)
        self.assertEqual(scoring[types.Boolean, types.Any], 1)
        self.assertEqual(scoring[types.Any, types.Boolean], 1)

    def test_select_and_sort_2(self):
        os = OverloadSelector()
        os.append(1, (types.Container,))
        os.append(2, (types.Sequence,))
        os.append(3, (types.MutableSequence,))
        os.append(4, (types.List,))
        compats = os._select_compatible((types.List,))
        self.assertEqual(len(compats), 4)
        ordered, scoring = os._sort_signatures(compats)
        self.assertEqual(len(ordered), 4)
        self.assertEqual(len(scoring), 4)
        self.assertEqual(ordered[0], (types.List,))
        self.assertEqual(scoring[types.List,], 0)
        self.assertEqual(scoring[types.MutableSequence,], 1)
        self.assertEqual(scoring[types.Sequence,], 2)
        self.assertEqual(scoring[types.Container,], 3)

    def test_match(self):
        os = OverloadSelector()
        self.assertTrue(os._match(formal=types.Boolean, actual=types.boolean))
        self.assertTrue(os._match(formal=types.Boolean, actual=types.Boolean))
        self.assertTrue(issubclass(types.Sequence, types.Container))
        self.assertTrue(os._match(formal=types.Container, actual=types.Sequence))
        self.assertFalse(os._match(formal=types.Sequence, actual=types.Container))
        self.assertTrue(os._match(formal=types.Any, actual=types.Any))
        self.assertTrue(os._match(formal=types.Any, actual=types.Container))
        self.assertFalse(os._match(formal=types.Container, actual=types.Any))

    def test_ambiguous_detection(self):
        os = OverloadSelector()
        os.append(1, (types.Any, types.Boolean))
        os.append(2, (types.Integer, types.Boolean))
        self.assertEqual(os.find((types.boolean, types.boolean)), 1)
        with self.assertRaises(NumbaNotImplementedError) as raises:
            os.find((types.boolean, types.int32))
        os.append(3, (types.Any, types.Any))
        self.assertEqual(os.find((types.boolean, types.int32)), 3)
        self.assertEqual(os.find((types.boolean, types.boolean)), 1)
        os.append(4, (types.Boolean, types.Any))
        with self.assertRaises(NumbaTypeError) as raises:
            os.find((types.boolean, types.boolean))
        self.assertIn('2 ambiguous signatures', str(raises.exception))
        os.append(5, (types.boolean, types.boolean))
        self.assertEqual(os.find((types.boolean, types.boolean)), 5)

    def test_subclass_specialization(self):
        os = OverloadSelector()
        self.assertTrue(issubclass(types.Sequence, types.Container))
        os.append(1, (types.Container, types.Container))
        lstty = types.List(types.boolean)
        self.assertEqual(os.find((lstty, lstty)), 1)
        os.append(2, (types.Container, types.Sequence))
        self.assertEqual(os.find((lstty, lstty)), 2)

    def test_cache(self):
        os = OverloadSelector()
        self.assertEqual(len(os._cache), 0)
        os.append(1, (types.Any,))
        self.assertEqual(os.find((types.int32,)), 1)
        self.assertEqual(len(os._cache), 1)
        os.append(2, (types.Integer,))
        self.assertEqual(len(os._cache), 0)
        self.assertEqual(os.find((types.int32,)), 2)
        self.assertEqual(len(os._cache), 1)
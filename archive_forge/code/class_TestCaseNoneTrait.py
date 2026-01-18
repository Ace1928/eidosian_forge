import unittest
from traits.api import HasTraits, TraitError
from traits.trait_types import _NoneTrait
class TestCaseNoneTrait(unittest.TestCase):

    def test_none(self):
        obj = A()
        self.assertIsNone(obj.none_atr)

    def test_assign_non_none(self):
        with self.assertRaises(TraitError):
            A(none_atr=5)

    def test_default_value_not_none(self):
        with self.assertRaises(ValueError):

            class TestClass(HasTraits):
                none_trait = _NoneTrait(default_value=[])
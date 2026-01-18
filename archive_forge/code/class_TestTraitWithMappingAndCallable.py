import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class TestTraitWithMappingAndCallable(unittest.TestCase):
    """ Test that demonstrates a usage of Trait where TraitMap is used but it
    cannot be replaced with Map. The callable causes the key value to be
    changed to match the mapped value.

    e.g. this would not work:

        value = Union(
            Map({"white": 0, "red": 1, (0,0,0): 999}),
            NewTraitType(),
            default_value="white",
        )

        where NewTraitType is a subclass of TraitType with ``validate`` simply
        calls str_cast_to_int
    """

    def test_trait_default(self):
        obj = TraitWithMappingAndCallable()
        self.assertEqual(obj.value, 5)
        self.assertEqual(obj.value_, 5)

    def test_trait_set_value_use_callable(self):
        obj = TraitWithMappingAndCallable(value='red')
        self.assertEqual(obj.value, 3)
        self.assertEqual(obj.value_, 3)

    def test_trait_set_value_use_mapping(self):
        obj = TraitWithMappingAndCallable(value=(0, 0, 0))
        self.assertEqual(obj.value, (0, 0, 0))
        self.assertEqual(obj.value_, 999)
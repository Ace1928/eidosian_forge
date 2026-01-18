from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
class TestAsCtrait(unittest.TestCase):

    def test_as_ctrait_from_ctrait(self):
        ct = Int().as_ctrait()
        result = as_ctrait(ct)
        self.assertIs(result, ct)

    def test_as_ctrait_from_class(self):
        result = as_ctrait(Int)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, Int)

    def test_as_ctrait_from_instance(self):
        trait = Int()
        result = as_ctrait(trait)
        self.assertIsInstance(result, CTrait)
        self.assertIs(result.handler, trait)

    def test_as_ctrait_from_trait_factory(self):
        int_trait_factory = TraitFactory(lambda: Int().as_ctrait())
        with reset_trait_factory():
            result = as_ctrait(int_trait_factory)
            ct = int_trait_factory.as_ctrait()
        self.assertIsInstance(result, CTrait)
        self.assertIs(result, ct)

    def test_as_ctrait_raise_exception(self):
        with self.assertRaises(TypeError):
            as_ctrait(1)
        with self.assertRaises(TypeError):
            as_ctrait(int)
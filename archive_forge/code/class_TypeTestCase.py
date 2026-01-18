import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
class TypeTestCase(unittest.TestCase):
    """ Tests the Type trait and its alias - the Subclass trait"""

    def test_type_base(self):
        model = ExampleTypeModel(_class=BaseClass)
        self.assertIsInstance(model._class(), BaseClass)

    def test_type_derived(self):
        model = ExampleTypeModel(_class=DerivedClass)
        self.assertIsInstance(model._class(), DerivedClass)

    def test_invalid_type(self):
        example_model = ExampleTypeModel()

        def assign_invalid():
            example_model._class = UnrelatedClass
        self.assertRaises(TraitError, assign_invalid)

    def test_subclass_base(self):
        model = ExampleSubclassModel(_class=BaseClass)
        self.assertIsInstance(model._class(), BaseClass)

    def test_subclass_derived(self):
        model = ExampleSubclassModel(_class=DerivedClass)
        self.assertIsInstance(model._class(), DerivedClass)

    def test_invalid_subclass(self):
        example_model = ExampleSubclassModel()

        def assign_invalid():
            example_model._class = UnrelatedClass
        self.assertRaises(TraitError, assign_invalid)
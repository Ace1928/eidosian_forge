import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class EnumTestCase(unittest.TestCase):

    def test_valid_enum(self):
        example_model = ExampleModel(root='model1')
        example_model.root = 'model2'

    def test_invalid_enum(self):
        example_model = ExampleModel(root='model1')

        def assign_invalid():
            example_model.root = 'not_valid_model'
        self.assertRaises(TraitError, assign_invalid)

    def test_enum_list(self):
        example = EnumListExample()
        self.assertEqual(example.value, 'foo')
        self.assertEqual(example.value_default, 'bar')
        self.assertEqual(example.value_name, 'foo')
        self.assertEqual(example.value_name_default, 'bar')
        example.value = 'bar'
        self.assertEqual(example.value, 'bar')
        with self.assertRaises(TraitError):
            example.value = 'something'
        with self.assertRaises(TraitError):
            example.value = 0
        example.values = ['one', 'two', 'three']
        example.value_name = 'two'
        self.assertEqual(example.value_name, 'two')
        with self.assertRaises(TraitError):
            example.value_name = 'bar'

    def test_enum_tuple(self):
        example = EnumTupleExample()
        self.assertEqual(example.value, 'foo')
        self.assertEqual(example.value_default, 'bar')
        self.assertEqual(example.value_name, 'foo')
        self.assertEqual(example.value_name_default, 'bar')
        example.value = 'bar'
        self.assertEqual(example.value, 'bar')
        with self.assertRaises(TraitError):
            example.value = 'something'
        with self.assertRaises(TraitError):
            example.value = 0
        example.values = ('one', 'two', 'three')
        example.value_name = 'two'
        self.assertEqual(example.value_name, 'two')
        with self.assertRaises(TraitError):
            example.value_name = 'bar'

    def test_enum_enum(self):
        example = EnumEnumExample()
        self.assertEqual(example.value, FooEnum.foo)
        self.assertEqual(example.value_default, FooEnum.bar)
        self.assertEqual(example.value_name, FooEnum.foo)
        self.assertEqual(example.value_name_default, FooEnum.bar)
        example.value = FooEnum.bar
        self.assertEqual(example.value, FooEnum.bar)
        with self.assertRaises(TraitError):
            example.value = 'foo'
        with self.assertRaises(TraitError):
            example.value = 0
        example.values = OtherEnum
        example.value_name = OtherEnum.two
        self.assertEqual(example.value_name, OtherEnum.two)
        with self.assertRaises(TraitError):
            example.value_name = FooEnum.bar

    def test_enum_collection(self):
        collection_enum = EnumCollectionExample()
        self.assertEqual('red', collection_enum.rgb)
        self.assertEqual('r', collection_enum.rgb_char)
        self.assertEqual('one', collection_enum.numbers)
        self.assertEqual('abcdefg', collection_enum.letters)
        self.assertEqual('yes', collection_enum.yes_no)
        self.assertEqual(0, collection_enum.digits)
        self.assertEqual(1, collection_enum.int_set_enum)
        self.assertEqual(1, collection_enum.two_digits)
        self.assertEqual(8, collection_enum.single_digit)
        collection_enum.rgb = 'blue'
        self.assertEqual('blue', collection_enum.rgb)
        collection_enum.rgb_char = 'g'
        self.assertEqual('g', collection_enum.rgb_char)
        collection_enum.yes_no = 'no'
        self.assertEqual('no', collection_enum.yes_no)
        for i in range(10):
            collection_enum.digits = i
            self.assertEqual(i, collection_enum.digits)
        collection_enum.two_digits = 2
        self.assertEqual(2, collection_enum.two_digits)
        with self.assertRaises(TraitError):
            collection_enum.rgb = 'two'
        with self.assertRaises(TraitError):
            collection_enum.letters = 'b'
        with self.assertRaises(TraitError):
            collection_enum.yes_no = 'n'
        with self.assertRaises(TraitError):
            collection_enum.digits = 10
        with self.assertRaises(TraitError):
            collection_enum.single_digit = 9
        with self.assertRaises(TraitError):
            collection_enum.single_digit = None
        with self.assertRaises(TraitError):
            collection_enum.int_set_enum = {1, 2}
        self.assertEqual(1, collection_enum.correct_int_set_enum)
        collection_enum.correct_int_set_enum = {1, 2}
        with self.assertRaises(TraitError):
            collection_enum.correct_int_set_enum = 20

    def test_empty_enum(self):
        with self.assertRaises(TraitError):

            class EmptyEnum(HasTraits):
                a = Enum()
            EmptyEnum()

    def test_too_many_arguments_for_dynamic_enum(self):
        with self.assertRaises(TraitError):
            Enum('red', 'green', values='values')

    def test_attributes(self):
        static_enum = Enum(1, 2, 3)
        self.assertEqual(static_enum.values, (1, 2, 3))
        self.assertIsNone(static_enum.name, None)
        dynamic_enum = Enum(values='values')
        self.assertIsNone(dynamic_enum.values)
        self.assertEqual(dynamic_enum.name, 'values')

    def test_explicit_collection_with_no_elements(self):
        with self.assertRaises(TraitError):
            Enum([])
        with self.assertRaises(TraitError):
            Enum(3.5, [])

    def test_base_enum(self):
        obj = EnumCollectionExample()
        self.assertEqual(obj.slow_enum, 'yes')
        obj.slow_enum = 'no'
        self.assertEqual(obj.slow_enum, 'no')
        with self.assertRaises(TraitError):
            obj.slow_enum = 'perhaps'
        self.assertEqual(obj.slow_enum, 'no')

    def test_dynamic_enum_in_tuple(self):

        class HasEnumInTuple(HasTraits):
            months = List(Int, value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            year_and_month = Tuple(Int(), Enum(values='months'))
        model = HasEnumInTuple()
        model.year_and_month = (1974, 8)
        self.assertEqual(model.year_and_month, (1974, 8))
        with self.assertRaises(TraitError):
            model.year_and_month = (1986, 13)

    def test_dynamic_enum_in_list(self):

        class HasEnumInList(HasTraits):
            digits = Set(Int)
            digit_sequence = List(Enum(values='digits'))
        model = HasEnumInList(digits={-1, 0, 1})
        model.digit_sequence = [-1, 0, 1, 1]
        with self.assertRaises(TraitError):
            model.digit_sequence = [-1, 0, 2, 1]
import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
class TestArrayOrNone(unittest.TestCase, UnittestTools):
    """
    Tests for the ArrayOrNone TraitType.

    """

    def test_default(self):
        foo = Foo()
        self.assertIsNone(foo.maybe_array)

    def test_explicit_default(self):
        foo = Foo()
        self.assertIsInstance(foo.maybe_array_with_default, numpy.ndarray)

    def test_default_validation(self):
        with self.assertRaises(TraitError):

            class Bar(HasTraits):
                bad_array = ArrayOrNone(shape=(None, None), value=[1, 2, 3])

    def test_setting_array_from_array(self):
        foo = Foo()
        test_array = numpy.arange(5)
        foo.maybe_array = test_array
        output_array = foo.maybe_array
        self.assertIsInstance(output_array, numpy.ndarray)
        self.assertEqual(output_array.dtype, test_array.dtype)
        self.assertEqual(output_array.shape, test_array.shape)
        self.assertTrue((output_array == test_array).all())

    def test_setting_array_from_list(self):
        foo = Foo()
        test_list = [5, 6, 7, 8, 9]
        foo.maybe_array = test_list
        output_array = foo.maybe_array
        self.assertIsInstance(output_array, numpy.ndarray)
        self.assertEqual(output_array.dtype, numpy.dtype(int))
        self.assertEqual(output_array.shape, (5,))
        self.assertTrue((output_array == test_list).all())

    def test_setting_array_from_none(self):
        foo = Foo()
        test_array = numpy.arange(5)
        self.assertIsNone(foo.maybe_array)
        foo.maybe_array = test_array
        self.assertIsInstance(foo.maybe_array, numpy.ndarray)
        foo.maybe_array = None
        self.assertIsNone(foo.maybe_array)

    def test_dtype(self):
        foo = Foo()
        foo.maybe_float_array = [1, 2, 3]
        array_value = foo.maybe_float_array
        self.assertIsInstance(array_value, numpy.ndarray)
        self.assertEqual(array_value.dtype, numpy.dtype(float))

    def test_shape(self):
        foo = Foo()
        with self.assertRaises(TraitError):
            foo.maybe_two_d_array = [1, 2, 3]

    def test_change_notifications(self):
        foo = Foo()
        test_array = numpy.arange(-7, -2)
        different_test_array = numpy.arange(10)
        with self.assertTraitDoesNotChange(foo, 'maybe_array'):
            foo.maybe_array = None
        with self.assertTraitChanges(foo, 'maybe_array'):
            foo.maybe_array = test_array
        with self.assertTraitDoesNotChange(foo, 'maybe_array'):
            foo.maybe_array = test_array
        with self.assertTraitChanges(foo, 'maybe_array'):
            foo.maybe_array = different_test_array
        different_test_array += 2
        with self.assertTraitDoesNotChange(foo, 'maybe_array'):
            foo.maybe_array = different_test_array
        with self.assertTraitChanges(foo, 'maybe_array'):
            foo.maybe_array = None

    def test_comparison_mode_override(self):
        foo = Foo()
        test_array = numpy.arange(-7, 2)
        with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
            foo.maybe_array_no_compare = None
        with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
            foo.maybe_array_no_compare = test_array
        with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
            foo.maybe_array_no_compare = test_array

    def test_default_value_copied(self):
        test_default = numpy.arange(100.0, 110.0)

        class FooBar(HasTraits):
            foo = ArrayOrNone(value=test_default)
            bar = ArrayOrNone(value=test_default)
        foo_bar = FooBar()
        self.assertTrue((foo_bar.foo == test_default).all())
        self.assertTrue((foo_bar.bar == test_default).all())
        test_default += 2.0
        self.assertFalse((foo_bar.foo == test_default).all())
        self.assertFalse((foo_bar.bar == test_default).all())
        foo = foo_bar.foo
        foo += 1729.0
        self.assertFalse((foo_bar.foo == foo_bar.bar).all())

    def test_safe_casting(self):

        class Bar(HasTraits):
            unsafe_f32 = ArrayOrNone(dtype='float32')
            safe_f32 = ArrayOrNone(dtype='float32', casting='safe')
        f64 = numpy.array([1], dtype='float64')
        f32 = numpy.array([1], dtype='float32')
        b = Bar()
        b.unsafe_f32 = f32
        b.unsafe_f32 = f64
        b.safe_f32 = f32
        with self.assertRaises(TraitError):
            b.safe_f32 = f64
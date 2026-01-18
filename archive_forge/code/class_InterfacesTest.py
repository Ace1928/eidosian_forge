import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class InterfacesTest(unittest.TestCase):

    def setUp(self):
        reset_global_adaptation_manager()
        register_factory(SampleListAdapter, Sample, IList)
        register_factory(ListAverageAdapter, IList, IAverage)
        register_factory(SampleFooAdapter, Sample, IFoo)
        register_factory(FooPlusAdapter, IFoo, IFooPlus)

    def test_provides_none(self):

        @provides()
        class Test(HasTraits):
            pass

    def test_provides_one(self):

        @provides(IFoo)
        class Test(HasTraits):

            def get_foo(self):
                return 'foo_and_only_foo'

            def get_average(self):
                return 42
        test = Test()
        self.assertIsInstance(test, IFoo)
        self.assertNotIsInstance(test, IAverage)

    def test_provides_multi(self):

        @provides(IFoo, IAverage, IList)
        class Test(HasTraits):

            def get_foo(self):
                return 'test_foo'

            def get_average(self):
                return 42

            def get_list(self):
                return [42]
        test = Test()
        self.assertIsInstance(test, IFoo)
        self.assertIsInstance(test, IAverage)
        self.assertIsInstance(test, IList)

    def test_provides_extended(self):
        """ Ensure that subclasses of Interfaces imply the superinterface.
        """

        @provides(IFooPlus)
        class Test(HasTraits):

            def get_foo(self):
                return 'some_test_foo'

            def get_foo_plus(self):
                return 'more_test_foo'
        test = Test()
        self.assertIsInstance(test, IFoo)
        self.assertIsInstance(test, IFooPlus)
        ta = TraitsHolder()
        ta.foo_adapted_to = test
        self.assertIs(ta.foo_adapted_to, test)

    def test_provides_bad(self):
        with self.assertRaises(Exception):

            @provides(Sample)
            class Test(HasTraits):
                pass

    def test_provides_with_no_interface_check(self):

        class Test(HasTraits):
            pass
        provides_ifoo = provides(IFoo)
        with self.set_check_interfaces(0):
            Test = provides_ifoo(Test)
        test = Test()
        self.assertIsInstance(test, IFoo)

    def test_provides_with_interface_check_warn(self):

        class Test(HasTraits):
            pass
        provides_ifoo = provides(IFoo)
        with self.set_check_interfaces(1):
            with self.assertWarns(DeprecationWarning) as warnings_cm:
                with self.assertLogs('traits', logging.WARNING):
                    Test = provides_ifoo(Test)
        test = Test()
        self.assertIsInstance(test, IFoo)
        self.assertIn('the @provides decorator will not perform interface checks', str(warnings_cm.warning))
        _, _, this_module = __name__.rpartition('.')
        self.assertIn(this_module, warnings_cm.filename)

    def test_provides_with_interface_check_error(self):

        class Test(HasTraits):
            pass
        provides_ifoo = provides(IFoo)
        with self.set_check_interfaces(2):
            with self.assertWarns(DeprecationWarning) as warnings_cm:
                with self.assertRaises(InterfaceError):
                    Test = provides_ifoo(Test)
        test = Test()
        self.assertIsInstance(test, IFoo)
        self.assertIn('the @provides decorator will not perform interface checks', str(warnings_cm.warning))
        _, _, this_module = __name__.rpartition('.')
        self.assertIn(this_module, warnings_cm.filename)

    def test_instance_adapt_no(self):
        ta = TraitsHolder()
        try:
            ta.a_no = SampleAverage()
        except TraitError:
            self.fail('Setting instance of interface should not require adaptation')
        self.assertRaises(TraitError, ta.trait_set, a_no=SampleList())
        self.assertRaises(TraitError, ta.trait_set, a_no=Sample())
        self.assertRaises(TraitError, ta.trait_set, a_no=SampleBad())

    def test_instance_adapt_yes(self):
        ta = TraitsHolder()
        ta.a_yes = SampleAverage()
        self.assertEqual(ta.a_yes.get_average(), 200.0)
        self.assertIsInstance(ta.a_yes, SampleAverage)
        self.assertFalse(hasattr(ta, 'a_yes_'))
        ta.a_yes = SampleList()
        self.assertEqual(ta.a_yes.get_average(), 20.0)
        self.assertIsInstance(ta.a_yes, ListAverageAdapter)
        self.assertFalse(hasattr(ta, 'a_yes_'))
        ta.a_yes = Sample()
        self.assertEqual(ta.a_yes.get_average(), 2.0)
        self.assertIsInstance(ta.a_yes, ListAverageAdapter)
        self.assertFalse(hasattr(ta, 'a_yes_'))
        self.assertRaises(TraitError, ta.trait_set, a_yes=SampleBad())

    def test_instance_adapt_default(self):
        ta = TraitsHolder()
        ta.a_default = SampleAverage()
        self.assertEqual(ta.a_default.get_average(), 200.0)
        self.assertIsInstance(ta.a_default, SampleAverage)
        self.assertFalse(hasattr(ta, 'a_default_'))
        ta.a_default = SampleList()
        self.assertEqual(ta.a_default.get_average(), 20.0)
        self.assertIsInstance(ta.a_default, ListAverageAdapter)
        self.assertFalse(hasattr(ta, 'a_default_'))
        ta.a_default = Sample()
        self.assertEqual(ta.a_default.get_average(), 2.0)
        self.assertIsInstance(ta.a_default, ListAverageAdapter)
        self.assertFalse(hasattr(ta, 'a_default_'))
        ta.a_default = SampleBad()
        self.assertEqual(ta.a_default, None)
        self.assertFalse(hasattr(ta, 'a_default_'))

    def test_adapted_to(self):
        ta = TraitsHolder()
        ta.list_adapted_to = object = Sample()
        result = ta.list_adapted_to.get_list()
        self.assertEqual(len(result), 3)
        for n in [1, 2, 3]:
            self.assertIn(n, result)
        self.assertIsInstance(ta.list_adapted_to, SampleListAdapter)
        self.assertEqual(ta.list_adapted_to_, object)
        ta.foo_adapted_to = object = Sample()
        self.assertEqual(ta.foo_adapted_to.get_foo(), 6)
        self.assertIsInstance(ta.foo_adapted_to, SampleFooAdapter)
        self.assertEqual(ta.foo_adapted_to_, object)
        ta.foo_plus_adapted_to = object = Sample(s1=5, s2=10, s3=15)
        self.assertEqual(ta.foo_plus_adapted_to.get_foo(), 30)
        self.assertEqual(ta.foo_plus_adapted_to.get_foo_plus(), 31)
        self.assertIsInstance(ta.foo_plus_adapted_to, FooPlusAdapter)
        self.assertEqual(ta.foo_plus_adapted_to_, object)

    def test_adapts_to(self):
        ta = TraitsHolder()
        ta.list_adapts_to = object = Sample()
        self.assertEqual(ta.list_adapts_to, object)
        result = ta.list_adapts_to_.get_list()
        self.assertEqual(len(result), 3)
        for n in [1, 2, 3]:
            self.assertIn(n, result)
        self.assertIsInstance(ta.list_adapts_to_, SampleListAdapter)
        ta.foo_adapts_to = object = Sample()
        self.assertEqual(ta.foo_adapts_to, object)
        self.assertEqual(ta.foo_adapts_to_.get_foo(), 6)
        self.assertIsInstance(ta.foo_adapts_to_, SampleFooAdapter)
        ta.foo_plus_adapts_to = object = Sample(s1=5, s2=10, s3=15)
        self.assertEqual(ta.foo_plus_adapts_to, object)
        self.assertEqual(ta.foo_plus_adapts_to_.get_foo(), 30)
        self.assertEqual(ta.foo_plus_adapts_to_.get_foo_plus(), 31)
        self.assertIsInstance(ta.foo_plus_adapts_to_, FooPlusAdapter)

    def test_decorated_class_name_and_docstring(self):
        self.assertEqual(SampleList.__name__, 'SampleList')
        self.assertEqual(SampleList.__doc__, 'SampleList docstring.')

    def test_instance_requires_provides(self):
        ta = TraitsHolder()
        provider = UndeclaredAverageProvider()
        with self.assertRaises(TraitError):
            ta.a_no = provider

    @contextlib.contextmanager
    def set_check_interfaces(self, check_interfaces_value):
        """
        Context manager to temporarily set has_traits.CHECK_INTERFACES
        to the given value.

        Parameters
        ----------
        check_interfaces_value : int
            One of 0 (don't check), 1 (check and log a warning on interface
            mismatch) or 2 (check and raise on interface mismatch).

        Returns
        -------
        context manager
        """
        old_check_interfaces = has_traits.CHECK_INTERFACES
        has_traits.CHECK_INTERFACES = check_interfaces_value
        try:
            yield
        finally:
            has_traits.CHECK_INTERFACES = old_check_interfaces
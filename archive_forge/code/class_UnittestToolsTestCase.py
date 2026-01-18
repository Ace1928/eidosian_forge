import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
class UnittestToolsTestCase(unittest.TestCase, UnittestTools):

    def setUp(self):
        self.test_object = TestObject()

    def test_when_using_with(self):
        """ Check normal use cases as a context manager.
        """
        test_object = self.test_object
        with self.assertTraitDoesNotChange(test_object, 'number') as result:
            test_object.flag = True
            test_object.number = 2.0
        msg = 'The assertion result is not None: {0}'.format(result.event)
        self.assertIsNone(result.event, msg=msg)
        with self.assertTraitChanges(test_object, 'number') as result:
            test_object.flag = False
            test_object.number = 5.0
        expected = (test_object, 'number', 2.0, 5.0)
        self.assertSequenceEqual(expected, result.event)
        with self.assertTraitChanges(test_object, 'number', count=2) as result:
            test_object.flag = False
            test_object.number = 4.0
            test_object.number = 3.0
        expected = [(test_object, 'number', 5.0, 4.0), (test_object, 'number', 4.0, 3.0)]
        self.assertSequenceEqual(expected, result.events)
        self.assertSequenceEqual(expected[-1], result.event)
        with self.assertTraitChanges(test_object, 'number') as result:
            test_object.flag = True
            test_object.add_to_number(10.0)
        expected = (test_object, 'number', 3.0, 13.0)
        self.assertSequenceEqual(expected, result.event)
        with self.assertTraitChanges(test_object, 'number', count=3) as result:
            test_object.flag = True
            test_object.add_to_number(10.0)
            test_object.add_to_number(10.0)
            test_object.add_to_number(10.0)
        expected = [(test_object, 'number', 13.0, 23.0), (test_object, 'number', 23.0, 33.0), (test_object, 'number', 33.0, 43.0)]
        self.assertSequenceEqual(expected, result.events)
        self.assertSequenceEqual(expected[-1], result.event)

    def test_assert_multi_changes(self):
        test_object = self.test_object
        with self.assertMultiTraitChanges([test_object], [], ['flag', 'number', 'list_of_numbers[]']) as results:
            test_object.number = 2.0
        events = list(filter(bool, (result.event for result in results)))
        msg = 'The assertion result is not None: {0}'.format(', '.join(events))
        self.assertFalse(events, msg=msg)
        with self.assertMultiTraitChanges([test_object], ['number', 'list_of_numbers[]'], ['flag']) as results:
            test_object.number = 5.0
        events = list(filter(bool, (result.event for result in results)))
        msg = 'The assertion result is None'
        self.assertTrue(events, msg=msg)

    def test_when_using_functions(self):
        test_object = self.test_object
        self.assertTraitChanges(test_object, 'number', 1, test_object.add_to_number, 13.0)
        self.assertTraitDoesNotChange(test_object, 'flag', test_object.add_to_number, 13.0)

    def test_indirect_events(self):
        """ Check catching indirect change events.
        """
        test_object = self.test_object
        with self.assertTraitChanges(test_object, 'list_of_numbers[]') as result:
            test_object.flag = True
            test_object.number = -3.0
        expected = (test_object, 'list_of_numbers_items', [], [-3.0])
        self.assertSequenceEqual(expected, result.event)

    def test_exception_inside_context(self):
        """ Check that exception inside the context statement block are
        propagated.

        """
        test_object = self.test_object
        with self.assertRaises(AttributeError):
            with self.assertTraitChanges(test_object, 'number'):
                test_object.i_do_exist
        with self.assertRaises(AttributeError):
            with self.assertTraitDoesNotChange(test_object, 'number'):
                test_object.i_do_exist

    def test_non_change_on_failure(self):
        """ Check behaviour when assertion should be raised for non trait
        change.

        """
        test_object = self.test_object
        traits = 'flag, number'
        with self.assertRaises(AssertionError):
            with self.assertTraitDoesNotChange(test_object, traits) as result:
                test_object.flag = True
                test_object.number = -3.0
        expected = [(test_object, 'flag', False, True), (test_object, 'number', 2.0, -3.0)]
        self.assertEqual(result.events, expected)

    def test_change_on_failure(self):
        """ Check behaviour when assertion should be raised for trait change.
        """
        test_object = self.test_object
        with self.assertRaises(AssertionError):
            with self.assertTraitChanges(test_object, 'number') as result:
                test_object.flag = True
        self.assertEqual(result.events, [])
        with self.assertRaises(AssertionError):
            with self.assertTraitChanges(test_object, 'number', count=3) as result:
                test_object.flag = True
                test_object.add_to_number(10.0)
                test_object.add_to_number(10.0)
        expected = [(test_object, 'number', 2.0, 12.0), (test_object, 'number', 12.0, 22.0)]
        self.assertSequenceEqual(expected, result.events)

    def test_asserts_in_context_block(self):
        """ Make sure that the traits context manager does not stop
        regular assertions inside the managed code block from happening.
        """
        test_object = TestObject(number=16.0)
        with self.assertTraitDoesNotChange(test_object, 'number'):
            self.assertEqual(test_object.number, 16.0)
        with self.assertRaisesRegex(AssertionError, '16\\.0 != 12\\.0'):
            with self.assertTraitDoesNotChange(test_object, 'number'):
                self.assertEqual(test_object.number, 12.0)

    def test_special_case_for_count(self):
        """ Count equal to 0 should be valid but it is discouraged.
        """
        test_object = TestObject(number=16.0)
        with self.assertTraitChanges(test_object, 'number', count=0):
            test_object.flag = True

    def test_assert_trait_changes_async(self):
        thread_count = 10
        events_per_thread = 1000

        class A(HasTraits):
            event = Event
        a = A()

        def thread_target(obj, count):
            """Fire obj.event 'count' times."""
            for _ in range(count):
                obj.event = True
        threads = [threading.Thread(target=thread_target, args=(a, events_per_thread)) for _ in range(thread_count)]
        expected_count = thread_count * events_per_thread
        with self.assertTraitChangesAsync(a, 'event', expected_count, timeout=60.0):
            for t in threads:
                t.start()
        for t in threads:
            t.join()

    def test_assert_trait_changes_async_events(self):
        thread_count = 10
        events_per_thread = 100

        class A(HasTraits):
            event = Event(Int)
        a = A()

        def thread_target(obj, count):
            """Fire obj.event 'count' times."""
            for n in range(count):
                time.sleep(0.001)
                obj.event = n
        threads = [threading.Thread(target=thread_target, args=(a, events_per_thread)) for _ in range(thread_count)]
        expected_count = thread_count * events_per_thread
        with self.assertTraitChangesAsync(a, 'event', expected_count, timeout=60.0) as event_collector:
            for t in threads:
                t.start()
        for t in threads:
            t.join()
        self.assertCountEqual(event_collector.events, list(range(events_per_thread)) * thread_count)

    def test_assert_trait_changes_async_failure(self):
        thread_count = 10
        events_per_thread = 10000

        class A(HasTraits):
            event = Event
        a = A()

        def thread_target(obj, count):
            """Fire obj.event 'count' times."""
            for _ in range(count):
                obj.event = True
        threads = [threading.Thread(target=thread_target, args=(a, events_per_thread)) for _ in range(thread_count)]
        expected_count = thread_count * events_per_thread
        with self.assertRaises(AssertionError):
            with self.assertTraitChangesAsync(a, 'event', expected_count + 1):
                for t in threads:
                    t.start()
        for t in threads:
            t.join()

    def test_assert_eventually_true_fails_on_timeout(self):

        class A(HasTraits):
            foo = Bool(False)
        a = A()

        def condition(a_object):
            return a_object.foo
        with self.assertRaises(self.failureException):
            self.assertEventuallyTrue(condition=condition, obj=a, trait='foo', timeout=1.0)

    def test_assert_eventually_true_passes_when_condition_becomes_true(self):

        class A(HasTraits):
            foo = Bool(False)

        def condition(a_object):
            return a_object.foo
        a = A()

        def thread_target(a):
            time.sleep(1.0)
            a.foo = True
        t = threading.Thread(target=thread_target, args=(a,))
        t.start()
        self.assertEventuallyTrue(condition=condition, obj=a, trait='foo', timeout=10.0)
        t.join()

    def test_assert_eventually_true_passes_when_condition_starts_true(self):

        class A(HasTraits):
            foo = Bool(True)

        def condition(a_object):
            return a_object.foo
        a = A()
        self.assertEventuallyTrue(condition=condition, obj=a, trait='foo', timeout=10.0)

    def test_assert_deprecated(self):
        with self.assertDeprecated():
            old_and_dull()

    def test_assert_deprecated_failures(self):
        with self.assertRaises(self.failureException):
            with self.assertDeprecated():
                pass

    def test_assert_deprecated_when_warning_already_issued(self):

        def old_and_dull_caller():
            old_and_dull()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', DeprecationWarning)
            old_and_dull_caller()
            with self.assertDeprecated():
                old_and_dull_caller()

    def test_assert_not_deprecated_failures(self):
        with self.assertRaises(self.failureException):
            with self.assertNotDeprecated():
                old_and_dull()

    def test_assert_not_deprecated(self):
        with self.assertNotDeprecated():
            pass

    def test_assert_not_deprecated_when_warning_already_issued(self):

        def old_and_dull_caller():
            old_and_dull()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', DeprecationWarning)
            old_and_dull_caller()
            with self.assertRaises(self.failureException):
                with self.assertNotDeprecated():
                    old_and_dull_caller()

    def test__catch_warnings_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            with self._catch_warnings():
                pass
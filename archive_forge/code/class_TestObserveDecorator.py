import unittest
from traits.api import (
from traits.observation.api import (
class TestObserveDecorator(unittest.TestCase):
    """ General tests for the observe decorator. """

    def test_warning_on_handler_with_bad_signature(self):
        message_regex = 'should be callable with a single positional argument'
        with self.assertWarnsRegex(UserWarning, message_regex):

            class A(HasTraits):
                foo = Int()

                @observe('foo')
                def _do_something_when_foo_changes(self):
                    pass
        with self.assertWarnsRegex(UserWarning, message_regex):

            class B(HasTraits):
                foo = Int()

                @observe('foo')
                def _do_something_when_foo_changes(self, **kwargs):
                    pass

    def test_decorated_method_signatures(self):

        class A(HasTraits):
            foo = Int()
            call_count = Int(0)

            @observe('foo')
            def _the_usual_signature(self, event):
                self.call_count += 1

            @observe('foo')
            def _method_with_extra_optional_args(self, event, frombicate=True):
                self.call_count += 1

            @observe('foo')
            def _method_with_star_args(self, *args):
                self.call_count += 1

            @observe('foo')
            def _method_with_alternative_name(self, foo_change_event):
                self.call_count += 1

            @observe('foo')
            def _optional_second_argument(self, event=None):
                self.call_count += 1
        a = A()
        self.assertEqual(a.call_count, 0)
        a.foo = 23
        self.assertEqual(a.call_count, 5)
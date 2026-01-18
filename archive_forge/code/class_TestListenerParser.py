from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
class TestListenerParser(unittest.TestCase):

    def setUp(self):
        push_exception_handler(handler=lambda *args: None, reraise_exceptions=True)
        self.addTypeEqualityFunc(traits_listener.ListenerItem, partial(assert_listener_item_equal, self))

    def tearDown(self):
        pop_exception_handler()

    def test_listener_parser_single_string(self):
        text = 'some_trait_name'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='some_trait_name', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_listener_parser_trait_of_trait_dot(self):
        text = 'parent.child'
        parser = traits_listener.ListenerParser(text=text)
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER)
        expected_child = traits_listener.ListenerItem(name='child', next=None, **common_traits)
        expected_parent = traits_listener.ListenerItem(name='parent', next=expected_child, **common_traits)
        self.assertEqual(parser.listener, expected_parent)

    def test_listener_parser_trait_of_trait_of_trait_mixed(self):
        text = 'parent.child1:child2'
        parser = traits_listener.ListenerParser(text=text)
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', is_list_handler=False, type=traits_listener.ANY_LISTENER)
        expected_child2 = traits_listener.ListenerItem(name='child2', notify=True, next=None, **common_traits)
        expected_child1 = traits_listener.ListenerItem(name='child1', notify=False, next=expected_child2, **common_traits)
        expected_parent = traits_listener.ListenerItem(name='parent', notify=True, next=expected_child1, **common_traits)
        self.assertEqual(parser.listener, expected_parent)

    def test_parse_comma_separated_text(self):
        text = 'child1, child2, child3'
        parser = traits_listener.ListenerParser(text=text)
        listener_group = parser.listener
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        expected_items = [traits_listener.ListenerItem(name='child1', **common_traits), traits_listener.ListenerItem(name='child2', **common_traits), traits_listener.ListenerItem(name='child3', **common_traits)]
        self.assertEqual(len(listener_group.items), len(expected_items))
        for actual, expected in zip(listener_group.items, expected_items):
            self.assertEqual(actual, expected)

    def test_parse_comma_separated_text_trailing_comma(self):
        text = 'child1, child2, child3,'
        with self.assertRaises(TraitError):
            traits_listener.ListenerParser(text=text)

    def test_parse_text_with_question_mark(self):
        text = 'foo?.bar?'
        parser = traits_listener.ListenerParser(text=text)
        listener = parser.listener
        self.assertEqual(listener.name, 'foo?')
        listener = listener.next
        self.assertEqual(listener.name, 'bar?')

    def test_parse_nested_empty_prefix_with_question_mark(self):
        text = 'foo.?'
        with self.assertRaises(TraitError) as exception_context:
            traits_listener.ListenerParser(text=text)
        self.assertIn('Expected non-empty name', str(exception_context.exception))

    def test_parse_question_mark_only(self):
        text = '?'
        with self.assertRaises(TraitError) as exception_context:
            traits_listener.ListenerParser(text=text)
        self.assertIn('Expected non-empty name', str(exception_context.exception))

    def test_parse_with_asterisk(self):
        text = 'prefix*'
        parser = traits_listener.ListenerParser(text=text)
        actual = parser.listener
        expected = traits_listener.ListenerItem(name='prefix', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=actual)
        self.assertEqual(actual, expected)

    def test_parse_text_with_metadata(self):
        text = 'prefix+foo'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='prefix*', metadata_name='foo', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_parse_is_anytrait_plus(self):
        text = '+'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='*', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_parse_is_anytrait_minus(self):
        text = '-'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='*', metadata_name='', metadata_defined=False, is_anytrait=True, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_parse_nested_exclude_empty_metadata_name(self):
        text = 'foo-'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='foo*', metadata_name='', metadata_defined=False, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_parse_exclude_metadata(self):
        text = '-foo'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='*', metadata_name='foo', metadata_defined=False, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_parse_square_bracket(self):
        text = '[foo, bar]'
        parser = traits_listener.ListenerParser(text=text)
        listener_group = parser.listener
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        expected_items = [traits_listener.ListenerItem(name='foo', **common_traits), traits_listener.ListenerItem(name='bar', **common_traits)]
        self.assertEqual(len(listener_group.items), len(expected_items))
        for actual, expected in zip(listener_group.items, expected_items):
            self.assertEqual(actual, expected)

    def test_parse_square_bracket_nested_attribute(self):
        text = '[foo, bar].baz'
        parser = traits_listener.ListenerParser(text=text)
        listener_group = parser.listener
        self.assertEqual(len(listener_group.items), 2)
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER)
        child_listener = traits_listener.ListenerItem(name='baz', next=None, **common_traits)
        expected_items = [traits_listener.ListenerItem(name='foo', next=child_listener, **common_traits), traits_listener.ListenerItem(name='bar', next=child_listener, **common_traits)]
        self.assertEqual(len(listener_group.items), len(expected_items))
        for actual, expected in zip(listener_group.items, expected_items):
            self.assertEqual(actual, expected)

    def test_parse_square_bracket_in_middle(self):
        text = 'foo.[bar, baz]'
        parser = traits_listener.ListenerParser(text=text)
        actual_foo = parser.listener
        actual_next = actual_foo.next
        actual_foo.next = None
        common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
        expected_foo = traits_listener.ListenerItem(name='foo', **common_traits)
        self.assertEqual(actual_foo, expected_foo)
        expected_items = [traits_listener.ListenerItem(name='bar', **common_traits), traits_listener.ListenerItem(name='baz', **common_traits)]
        self.assertEqual(len(actual_next.items), len(expected_items))
        for actual, expected in zip(actual_next.items, expected_items):
            self.assertEqual(actual, expected)

    def test_parse_is_list_handler(self):
        text = 'foo[]'
        parser = traits_listener.ListenerParser(text=text)
        expected = traits_listener.ListenerItem(name='foo', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=True, type=traits_listener.ANY_LISTENER, next=None)
        self.assertEqual(parser.listener, expected)

    def test_listener_handler_for_method(self):

        class A:

            def __init__(self, value):
                self.value = value

            def square(self):
                return self.value * self.value
        a = A(7)
        listener_handler = traits_listener.ListenerHandler(a.square)
        handler = listener_handler()
        self.assertEqual(handler(), 49)
        del a, handler
        handler = listener_handler()
        self.assertEqual(handler, Undefined)

    def test_listener_handler_for_function(self):

        def square(value):
            return value * value
        listener_handler = traits_listener.ListenerHandler(square)
        handler = listener_handler()
        self.assertEqual(handler(9), 81)
        del square, handler
        handler = listener_handler()
        self.assertEqual(handler(5), 25)
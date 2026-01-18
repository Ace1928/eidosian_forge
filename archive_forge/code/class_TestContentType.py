from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
class TestContentType(TestCase):

    def test___init___None_errors(self):
        raises_value_error = Raises(MatchesException(ValueError))
        self.assertThat(lambda: ContentType(None, None), raises_value_error)
        self.assertThat(lambda: ContentType(None, 'traceback'), raises_value_error)
        self.assertThat(lambda: ContentType('text', None), raises_value_error)

    def test___init___sets_ivars(self):
        content_type = ContentType('foo', 'bar')
        self.assertEqual('foo', content_type.type)
        self.assertEqual('bar', content_type.subtype)
        self.assertEqual({}, content_type.parameters)

    def test___init___with_parameters(self):
        content_type = ContentType('foo', 'bar', {'quux': 'thing'})
        self.assertEqual({'quux': 'thing'}, content_type.parameters)

    def test___eq__(self):
        content_type1 = ContentType('foo', 'bar', {'quux': 'thing'})
        content_type2 = ContentType('foo', 'bar', {'quux': 'thing'})
        content_type3 = ContentType('foo', 'bar', {'quux': 'thing2'})
        self.assertTrue(content_type1.__eq__(content_type2))
        self.assertFalse(content_type1.__eq__(content_type3))

    def test_basic_repr(self):
        content_type = ContentType('text', 'plain')
        self.assertThat(repr(content_type), Equals('text/plain'))

    def test_extended_repr(self):
        content_type = ContentType('text', 'plain', {'foo': 'bar', 'baz': 'qux'})
        self.assertThat(repr(content_type), Equals('text/plain; baz="qux"; foo="bar"'))
import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
class UnwrapText(unittest.TestCase):

    def _unwrap(self, select):
        return _transform(FOO, Transformer(select).unwrap())

    def test_unwrap_element(self):
        self.assertEqual(self._unwrap('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (INSIDE, TEXT, u'FOO'), (None, END, u'root')])

    def test_unwrap_text(self):
        self.assertEqual(self._unwrap('foo/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_unwrap_attr(self):
        self.assertEqual(self._unwrap('foo/@name'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ATTR, ATTR, {'name': u'foo'}), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_unwrap_adjacent(self):
        self.assertEqual(_transform(FOOBAR, Transformer('foo|bar').unwrap()), [(None, START, u'root'), (None, TEXT, u'ROOT'), (INSIDE, TEXT, u'FOO'), (INSIDE, TEXT, u'BAR'), (None, END, u'root')])

    def test_unwrap_root(self):
        self.assertEqual(self._unwrap('.'), [(INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo')])

    def test_unwrap_text_root(self):
        self.assertEqual(_transform('foo', Transformer('.').unwrap()), [(OUTSIDE, TEXT, 'foo')])
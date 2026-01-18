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
class EmptyTest(unittest.TestCase):

    def _empty(self, select):
        return _transform(FOO, Transformer(select).empty())

    def test_empty_element(self):
        self.assertEqual(self._empty('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (EXIT, END, u'foo'), (None, END, u'root')])

    def test_empty_text(self):
        self.assertEqual(self._empty('foo/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_empty_attr(self):
        self.assertEqual(self._empty('foo/@name'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ATTR, ATTR, {'name': u'foo'}), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_empty_context(self):
        self.assertEqual(self._empty('.'), [(ENTER, START, u'root'), (EXIT, END, u'root')])

    def test_empty_text_context(self):
        self.assertEqual(_simplify(Transformer('.')(HTML(u'foo'), keep_marks=True)), [(OUTSIDE, TEXT, u'foo')])
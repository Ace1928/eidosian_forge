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
class WrapTest(unittest.TestCase):

    def _wrap(self, select, wrap='wrap'):
        return _transform(FOO, Transformer(select).wrap(wrap))

    def test_wrap_element(self):
        self.assertEqual(self._wrap('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'wrap'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, END, u'wrap'), (None, END, u'root')])

    def test_wrap_adjacent_elements(self):
        self.assertEqual(_transform(FOOBAR, Transformer('foo|bar').wrap('wrap')), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'wrap'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, END, u'wrap'), (None, START, u'wrap'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, END, u'wrap'), (None, END, u'root')])

    def test_wrap_text(self):
        self.assertEqual(self._wrap('foo/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (None, START, u'wrap'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'wrap'), (None, END, u'foo'), (None, END, u'root')])

    def test_wrap_root(self):
        self.assertEqual(self._wrap('.'), [(None, START, u'wrap'), (ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (EXIT, END, u'root'), (None, END, u'wrap')])

    def test_wrap_text_root(self):
        self.assertEqual(_transform('foo', Transformer('.').wrap('wrap')), [(None, START, u'wrap'), (OUTSIDE, TEXT, u'foo'), (None, END, u'wrap')])

    def test_wrap_with_element(self):
        element = Element('a', href='http://localhost')
        self.assertEqual(_transform('foo', Transformer('.').wrap(element), with_attrs=True), [(None, START, (u'a', {u'href': u'http://localhost'})), (OUTSIDE, TEXT, u'foo'), (None, END, u'a')])
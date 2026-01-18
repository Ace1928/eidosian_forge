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
class ReplaceTest(unittest.TestCase, ContentTestMixin):
    operation = 'replace'

    def test_replace_element(self):
        self.assertEqual(self._apply('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, TEXT, u'CONTENT 1'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_replace_text(self):
        self.assertEqual(self._apply('text()'), [(None, START, u'root'), (None, TEXT, u'CONTENT 1'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_replace_context(self):
        self.assertEqual(self._apply('.'), [(None, TEXT, u'CONTENT 1')])

    def test_replace_text_context(self):
        self.assertEqual(self._apply('.', html='foo'), [(None, TEXT, u'CONTENT 1')])

    def test_replace_adjacent_elements(self):
        self.assertEqual(self._apply('*'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, TEXT, u'CONTENT 1'), (None, TEXT, u'CONTENT 2'), (None, END, u'root')])

    def test_replace_all(self):
        self.assertEqual(self._apply('*|text()'), [(None, START, u'root'), (None, TEXT, u'CONTENT 1'), (None, TEXT, u'CONTENT 2'), (None, TEXT, u'CONTENT 3'), (None, END, u'root')])

    def test_replace_with_callback(self):
        count = [0]

        def content():
            count[0] += 1
            yield ('%2i.' % count[0])
        self.assertEqual(self._apply('*', content), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, TEXT, u' 1.'), (None, TEXT, u' 2.'), (None, END, u'root')])
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
class RemoveTest(unittest.TestCase):

    def _remove(self, select):
        return _transform(FOO, Transformer(select).remove())

    def test_remove_element(self):
        self.assertEqual(self._remove('foo|bar'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, END, u'root')])

    def test_remove_text(self):
        self.assertEqual(self._remove('//text()'), [(None, START, u'root'), (None, START, u'foo'), (None, END, u'foo'), (None, END, u'root')])

    def test_remove_attr(self):
        self.assertEqual(self._remove('foo/@name'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_remove_context(self):
        self.assertEqual(self._remove('.'), [])

    def test_remove_text_context(self):
        self.assertEqual(_transform('foo', Transformer('.').remove()), [])
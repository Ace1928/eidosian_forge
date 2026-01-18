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
class CopyTest(unittest.TestCase, BufferTestMixin):
    operation = 'copy'

    def test_copy_element(self):
        self.assertEqual(self._apply('foo')[1], [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')]])

    def test_copy_adjacent_elements(self):
        self.assertEqual(self._apply('foo|bar')[1], [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')], [(None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar')]])

    def test_copy_all(self):
        self.assertEqual(self._apply('*|text()')[1], [[(None, TEXT, u'ROOT')], [(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')], [(None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar')]])

    def test_copy_text(self):
        self.assertEqual(self._apply('*/text()')[1], [[(None, TEXT, u'FOO')], [(None, TEXT, u'BAR')]])

    def test_copy_context(self):
        self.assertEqual(self._apply('.')[1], [[(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')]])

    def test_copy_attribute(self):
        self.assertEqual(self._apply('foo/@name', with_attrs=True)[1], [[(None, ATTR, {'name': u'foo'})]])

    def test_copy_attributes(self):
        self.assertEqual(self._apply('foo/@*', with_attrs=True)[1], [[(None, ATTR, {u'name': u'foo', u'size': u'100'})]])
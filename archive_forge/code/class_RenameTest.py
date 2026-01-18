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
class RenameTest(unittest.TestCase):

    def _rename(self, select):
        return _transform(FOOBAR, Transformer(select).rename('foobar'))

    def test_rename_root(self):
        self.assertEqual(self._rename('.'), [(ENTER, START, u'foobar'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (INSIDE, START, u'bar'), (INSIDE, TEXT, u'BAR'), (INSIDE, END, u'bar'), (EXIT, END, u'foobar')])

    def test_rename_element(self):
        self.assertEqual(self._rename('foo|bar'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foobar'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foobar'), (ENTER, START, u'foobar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'foobar'), (None, END, u'root')])

    def test_rename_text(self):
        self.assertEqual(self._rename('foo/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])
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
class MapTest(unittest.TestCase):

    def _map(self, select, kind=None):
        data = []

        def record(d):
            data.append(d)
            return d
        _transform(FOOBAR, Transformer(select).map(record, kind))
        return data

    def test_map_element(self):
        self.assertEqual(self._map('foo'), [(QName('foo'), Attrs([(QName('name'), u'foo'), (QName('size'), u'100')])), u'FOO', QName('foo')])

    def test_map_with_text_kind(self):
        self.assertEqual(self._map('.', TEXT), [u'ROOT', u'FOO', u'BAR'])

    def test_map_with_root_and_end_kind(self):
        self.assertEqual(self._map('.', END), [QName('foo'), QName('bar'), QName('root')])

    def test_map_with_attribute(self):
        self.assertEqual(self._map('foo/@name'), [(QName('foo@*'), Attrs([('name', u'foo')]))])
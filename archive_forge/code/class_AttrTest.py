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
class AttrTest(unittest.TestCase):

    def _attr(self, select, name, value):
        return _transform(FOOBAR, Transformer(select).attr(name, value), with_attrs=True)

    def test_set_existing_attr(self):
        self.assertEqual(self._attr('foo', 'name', 'FOO'), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'name': 'FOO', u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_set_new_attr(self):
        self.assertEqual(self._attr('foo', 'title', 'FOO'), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'name': u'foo', u'title': 'FOO', u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_attr_from_function(self):

        def set(name, event):
            self.assertEqual(name, 'name')
            return event[1][1].get('name').upper()
        self.assertEqual(self._attr('foo|bar', 'name', set), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'name': 'FOO', u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (ENTER, START, (u'bar', {u'name': 'BAR'})), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, END, u'root')])

    def test_remove_attr(self):
        self.assertEqual(self._attr('foo', 'name', None), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_remove_attr_with_function(self):

        def set(name, event):
            return None
        self.assertEqual(self._attr('foo', 'name', set), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])
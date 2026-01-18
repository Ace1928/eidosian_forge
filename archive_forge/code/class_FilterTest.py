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
class FilterTest(unittest.TestCase):

    def _filter(self, select, html=FOOBAR):
        """Returns a list of lists of filtered elements."""
        output = []

        def filtered(stream):
            interval = []
            output.append(interval)
            for event in stream:
                interval.append(event)
                yield event
        _transform(html, Transformer(select).filter(filtered))
        simplified = []
        for sub in output:
            simplified.append(_simplify([(None, event) for event in sub]))
        return simplified

    def test_filter_element(self):
        self.assertEqual(self._filter('foo'), [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')]])

    def test_filter_adjacent_elements(self):
        self.assertEqual(self._filter('foo|bar'), [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')], [(None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar')]])

    def test_filter_text(self):
        self.assertEqual(self._filter('*/text()'), [[(None, TEXT, u'FOO')], [(None, TEXT, u'BAR')]])

    def test_filter_root(self):
        self.assertEqual(self._filter('.'), [[(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')]])

    def test_filter_text_root(self):
        self.assertEqual(self._filter('.', 'foo'), [[(None, TEXT, u'foo')]])

    def test_filter_after_outside(self):
        stream = _transform('<root>x</root>', Transformer('//root/text()').filter(lambda x: x))
        self.assertEqual(list(stream), [(None, START, u'root'), (OUTSIDE, TEXT, u'x'), (None, END, u'root')])
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
def _map(self, select, kind=None):
    data = []

    def record(d):
        data.append(d)
        return d
    _transform(FOOBAR, Transformer(select).map(record, kind))
    return data
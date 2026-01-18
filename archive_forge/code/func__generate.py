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
def _generate():
    for mark, (kind, data, pos) in stream:
        if kind is START:
            if with_attrs:
                kv_attrs = dict(((six.text_type(k), v) for k, v in data[1]))
                data = (six.text_type(data[0]), kv_attrs)
            else:
                data = six.text_type(data[0])
        elif kind is END:
            data = six.text_type(data)
        elif kind is ATTR:
            kind = ATTR
            data = dict(((six.text_type(k), v) for k, v in data[1]))
        yield (mark, kind, data)
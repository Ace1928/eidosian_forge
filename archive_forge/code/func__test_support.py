import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def _test_support(self, strategy_class, text):
    path = PathParser(text, None, -1).parse()[0]
    return strategy_class.supports(path)
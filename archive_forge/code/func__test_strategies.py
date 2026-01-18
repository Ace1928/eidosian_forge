import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def _test_strategies(self, input, path, output, namespaces=None, variables=None):
    for strategy in self.strategies:
        if not strategy.supports(path):
            continue
        s = strategy(path)
        rendered = FakePath(s).select(input, namespaces=namespaces, variables=variables).render(encoding=None)
        msg = 'Bad render using %s strategy' % str(strategy)
        msg += '\nExpected:\t%r' % output
        msg += '\nRendered:\t%r' % rendered
        self.assertEqual(output, rendered, msg)
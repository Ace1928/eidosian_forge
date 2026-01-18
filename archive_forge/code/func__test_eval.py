import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def _test_eval(self, path, equiv=None, input=None, output='', namespaces=None, variables=None):
    path = Path(path)
    if equiv is not None:
        self.assertEqual(equiv, repr(path))
    if input is None:
        return
    rendered = path.select(input, namespaces=namespaces, variables=variables).render(encoding=None)
    msg = 'Bad output using whole path'
    msg += '\nExpected:\t%r' % output
    msg += '\nRendered:\t%r' % rendered
    self.assertEqual(output, rendered, msg)
    if len(path.paths) == 1:
        self._test_strategies(input, path.paths[0], output, namespaces=namespaces, variables=variables)
import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class AssertionNamesTests(unittest.SynchronousTestCase):
    """
    Tests for consistency of naming within TestCase assertion methods
    """

    def _getAsserts(self):
        dct = {}
        accumulateMethods(self, dct, 'assert')
        return [dct[k] for k in dct if not k.startswith('Not') and k != '_']

    def _name(self, x):
        return x.__name__

    def test_failUnlessMatchesAssert(self):
        """
        The C{failUnless*} test methods are a subset of the C{assert*} test
        methods.  This is intended to ensure that methods using the
        I{failUnless} naming scheme are not added without corresponding methods
        using the I{assert} naming scheme.  The I{assert} naming scheme is
        preferred, and new I{assert}-prefixed methods may be added without
        corresponding I{failUnless}-prefixed methods.
        """
        asserts = set(self._getAsserts())
        failUnlesses = set(prefixedMethods(self, 'failUnless'))
        self.assertEqual(failUnlesses, asserts.intersection(failUnlesses))

    def test_failIf_matches_assertNot(self):
        asserts = prefixedMethods(unittest.SynchronousTestCase, 'assertNot')
        failIfs = prefixedMethods(unittest.SynchronousTestCase, 'failIf')
        self.assertEqual(sorted(asserts, key=self._name), sorted(failIfs, key=self._name))

    def test_equalSpelling(self):
        for name, value in vars(self).items():
            if not callable(value):
                continue
            if name.endswith('Equal'):
                self.assertTrue(hasattr(self, name + 's'), f'{name} but no {name}s')
                self.assertEqual(value, getattr(self, name + 's'))
            if name.endswith('Equals'):
                self.assertTrue(hasattr(self, name[:-1]), f'{name} but no {name[:-1]}')
                self.assertEqual(value, getattr(self, name[:-1]))
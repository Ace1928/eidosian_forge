import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class FullyQualifiedNameTests(TestCase):
    """
    Test for L{fullyQualifiedName}.
    """

    def _checkFullyQualifiedName(self, obj, expected):
        """
        Helper to check that fully qualified name of C{obj} results to
        C{expected}.
        """
        self.assertEqual(fullyQualifiedName(obj), expected)

    def test_package(self):
        """
        L{fullyQualifiedName} returns the full name of a package and a
        subpackage.
        """
        import twisted
        self._checkFullyQualifiedName(twisted, 'twisted')
        import twisted.python
        self._checkFullyQualifiedName(twisted.python, 'twisted.python')

    def test_module(self):
        """
        L{fullyQualifiedName} returns the name of a module inside a package.
        """
        import twisted.python.compat
        self._checkFullyQualifiedName(twisted.python.compat, 'twisted.python.compat')

    def test_class(self):
        """
        L{fullyQualifiedName} returns the name of a class and its module.
        """
        self._checkFullyQualifiedName(FullyQualifiedNameTests, f'{__name__}.FullyQualifiedNameTests')

    def test_function(self):
        """
        L{fullyQualifiedName} returns the name of a function inside its module.
        """
        self._checkFullyQualifiedName(fullyQualifiedName, 'twisted.python.reflect.fullyQualifiedName')

    def test_boundMethod(self):
        """
        L{fullyQualifiedName} returns the name of a bound method inside its
        class and its module.
        """
        self._checkFullyQualifiedName(self.test_boundMethod, f'{__name__}.{self.__class__.__name__}.test_boundMethod')

    def test_unboundMethod(self):
        """
        L{fullyQualifiedName} returns the name of an unbound method inside its
        class and its module.
        """
        self._checkFullyQualifiedName(self.__class__.test_unboundMethod, f'{__name__}.{self.__class__.__name__}.test_unboundMethod')
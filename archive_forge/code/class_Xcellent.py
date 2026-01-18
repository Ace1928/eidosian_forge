from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@implementer(IAttrX)
class Xcellent:
    """
    L{IAttrX} implementation for test of adapter with C{__cmp__}.
    """

    def x(self):
        """
        Return a value.

        @return: a value
        """
        return 'x!'
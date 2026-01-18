from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@implementer(IProxiedSubInterface)
class Booable:
    """
    An implementation of IProxiedSubInterface
    """
    yayed = False
    booed = False

    def yay(self, *a, **kw):
        """
        Mark the fact that 'yay' has been called.
        """
        self.yayed = True

    def boo(self):
        """
        Mark the fact that 'boo' has been called.1
        """
        self.booed = True
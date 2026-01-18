from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@implementer(IProxiedInterface)
class Yayable:
    """
    A provider of L{IProxiedInterface} which increments a counter for
    every call to C{yay}.

    @ivar yays: The number of times C{yay} has been called.
    """

    def __init__(self):
        self.yays = 0
        self.yayArgs = []

    def yay(self, *a, **kw):
        """
        Increment C{self.yays}.
        """
        self.yays += 1
        self.yayArgs.append((a, kw))
        return self.yays
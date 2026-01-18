from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class EnhancedWrapper(YayableWrapper):
    """
            This class overrides the 'yay' method.
            """
    wrappedYays = 1

    def yay(self, *a, **k):
        self.wrappedYays += 1
        return YayableWrapper.yay(self, *a, **k) + 7
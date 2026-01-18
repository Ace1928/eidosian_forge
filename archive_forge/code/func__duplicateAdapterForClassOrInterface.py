from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def _duplicateAdapterForClassOrInterface(self, original):
    """
        Verify that L{components.registerAdapter} raises L{ValueError} if the
        from-type/interface and to-interface pair is not unique.
        """
    firstAdapter = lambda o: False
    secondAdapter = lambda o: True
    components.registerAdapter(firstAdapter, original, ITest)
    self.assertRaises(ValueError, components.registerAdapter, secondAdapter, original, ITest)
    self.assertIs(components.getAdapterFactory(original, ITest, None), firstAdapter)
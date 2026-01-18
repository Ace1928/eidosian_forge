from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class RegistryUsingMixin:
    """
    Mixin for test cases which modify the global registry somehow.
    """

    def setUp(self):
        """
        Configure L{twisted.python.components.registerAdapter} to mutate an
        alternate registry to improve test isolation.
        """
        scratchRegistry = AdapterRegistry()
        self.patch(components, 'globalRegistry', scratchRegistry)
        hook = _addHook(scratchRegistry)
        self.addCleanup(_removeHook, hook)
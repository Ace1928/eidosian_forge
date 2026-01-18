import unittest
from zope.interface.tests import OptimizationTestMixin
class VerifyingAdapterRegistryTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.adapter import VerifyingAdapterRegistry
        return VerifyingAdapterRegistry

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_verify_object_provides_IAdapterRegistry(self):
        from zope.interface.interfaces import IAdapterRegistry
        from zope.interface.verify import verifyObject
        registry = self._makeOne()
        verifyObject(IAdapterRegistry, registry)
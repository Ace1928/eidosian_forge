import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class TestPersistentComponentsDict(TestPersistentDictComponents):

    def _getTargetClass(self):
        return PersistentComponentsDict
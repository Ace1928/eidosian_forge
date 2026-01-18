import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class UnhashableComponentsTests(ComponentsTests):

    def _getTargetClass(self):

        class Components(super(UnhashableComponentsTests, self)._getTargetClass(), dict):
            pass
        return Components
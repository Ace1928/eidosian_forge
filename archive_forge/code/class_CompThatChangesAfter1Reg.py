import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class CompThatChangesAfter1Reg(self._getTargetClass()):
    reg_count = 0

    def registerUtility(self, *args):
        self.reg_count += 1
        super().registerUtility(*args)
        if self.reg_count == 1:
            self._utility_registrations = dict(self._utility_registrations)
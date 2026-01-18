import unittest
from zope.interface.tests import OptimizationTestMixin
class LookupBaseTests(LookupBaseFallbackTests, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.adapter import LookupBase
        return LookupBase
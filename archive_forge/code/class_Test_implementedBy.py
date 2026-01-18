import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_implementedBy(Test_implementedByFallback, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.declarations import implementedBy
        return implementedBy
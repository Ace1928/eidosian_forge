import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_providedBy(Test_providedByFallback, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.declarations import providedBy
        return providedBy
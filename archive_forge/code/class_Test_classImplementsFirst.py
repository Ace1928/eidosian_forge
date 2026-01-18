import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_classImplementsFirst(Test_classImplements):

    def _callFUT(self, cls, iface):
        from zope.interface.declarations import classImplementsFirst
        result = classImplementsFirst(cls, iface)
        self.assertIsNone(result)
        return cls

    def _order_for_two(self, applied_first, applied_second):
        return (applied_second, applied_first)
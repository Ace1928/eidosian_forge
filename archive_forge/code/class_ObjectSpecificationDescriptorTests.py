import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ObjectSpecificationDescriptorTests(ObjectSpecificationDescriptorFallbackTests, OptimizationTestMixin):

    def _getTargetClass(self):
        from zope.interface.declarations import ObjectSpecificationDescriptor
        return ObjectSpecificationDescriptor
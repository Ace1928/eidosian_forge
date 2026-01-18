import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class InterfaceBaseTests(InterfaceBaseTestsMixin, OptimizationTestMixin, unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.interface import InterfaceBase
        return InterfaceBase
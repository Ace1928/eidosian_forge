import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class StrictClassProvides(ClassProvides):

    def _do_calculate_ro(self, base_mros):
        return ClassProvides._do_calculate_ro(self, base_mros=base_mros, strict=True)
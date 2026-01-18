import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def _order_for_two(self, applied_first, applied_second):
    return (applied_second, applied_first)
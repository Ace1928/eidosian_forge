import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def _getEmpty(self):
    from zope.interface.declarations import _empty
    return _empty
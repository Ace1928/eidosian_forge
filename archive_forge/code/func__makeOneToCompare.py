import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def _makeOneToCompare(self):
    return self._makeOne('a', 'b')
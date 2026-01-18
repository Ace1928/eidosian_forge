import unittest
from zope.interface.tests import OptimizationTestMixin
def add_extendor(self, provided):
    self._extendors += (provided,)
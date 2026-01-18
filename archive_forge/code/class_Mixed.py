import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class Mixed(Left, Right):
    __implemented__ = (Left.__implemented__, Other.__implemented__)
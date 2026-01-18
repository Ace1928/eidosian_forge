import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class ICheckMe(Interface):
    attr = Attribute('My attr')

    def method():
        """A method"""
import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class IAdapt(Interface):

    @interfacemethod
    def __adapt__(self, obj):
        return 42
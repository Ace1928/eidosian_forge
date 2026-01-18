import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class IDerived2Adapt(IDerivedAdapt):
    """Overrides an inherited custom adapt."""

    @interfacemethod
    def __adapt__(self, obj):
        return 24
import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
@implementer(IAdapt)
class Conform24:

    def __conform__(self, iface):
        return 24
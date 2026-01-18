import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class IHaveADocString(Interface):
    """xxx"""
    __doc__ = Attribute('the doc')
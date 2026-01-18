import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class ITagMe(Interface):

    def method():
        """docstring"""
    method.optional = 1
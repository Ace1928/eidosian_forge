import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class CheckMe:
    __implemented__ = ICheckMe
    attr = 'value'

    def method(self):
        raise NotImplementedError()
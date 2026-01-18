import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class RaisesErrorOnModule(RaisesErrorOnMissing):

    def __init__(self):
        self.__name__ = 'foo'

    @property
    def __module__(self):
        raise AttributeError
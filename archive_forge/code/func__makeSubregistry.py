import unittest
from zope.interface.tests import OptimizationTestMixin
def _makeSubregistry(self, *provided):

    class Subregistry:

        def __init__(self):
            self._adapters = []
            self._subscribers = []
    return Subregistry()
import unittest
from zope.interface.tests import OptimizationTestMixin
class Subregistry:

    def __init__(self):
        self._adapters = []
        self._subscribers = []
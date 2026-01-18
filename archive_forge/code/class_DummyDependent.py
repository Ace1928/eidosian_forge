import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class DummyDependent:

    def __init__(self):
        self._changed = []

    def changed(self, originally_changed):
        self._changed.append(originally_changed)
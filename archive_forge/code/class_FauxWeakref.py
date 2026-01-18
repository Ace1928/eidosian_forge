import unittest
from zope.interface.tests import OptimizationTestMixin
class FauxWeakref:
    _unsub = None

    def __init__(self, here):
        self._here = here

    def __call__(self):
        return self if self._here else None

    def unsubscribe(self, target):
        self._unsub = target
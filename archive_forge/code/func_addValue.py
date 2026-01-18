import unittest
from zope.interface.tests import OptimizationTestMixin
def addValue(existing, new):
    existing = existing if existing is not None else CustomLeafSequence()
    existing.append(new)
    return existing
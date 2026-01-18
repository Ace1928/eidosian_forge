import unittest
from zope.interface.tests import OptimizationTestMixin
def _factory1(context):
    _called.setdefault('_factory1', []).append(context)
    return _exp1
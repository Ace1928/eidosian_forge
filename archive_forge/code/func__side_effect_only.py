import unittest
from zope.interface.tests import OptimizationTestMixin
def _side_effect_only(context):
    _called.setdefault('_side_effect_only', []).append(context)
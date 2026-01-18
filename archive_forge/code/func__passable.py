import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def _passable(*args, **kw):
    _passable_called_with.append((args, kw))
    return True
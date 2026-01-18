import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class ComparisonError:
    """
    An object which raises exceptions from its comparison methods.
    """

    def _error(self, other):
        raise ValueError('Comparison is broken')
    __eq__ = _error
    __ne__ = _error
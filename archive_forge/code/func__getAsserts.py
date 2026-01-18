import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def _getAsserts(self):
    dct = {}
    accumulateMethods(self, dct, 'assert')
    return [dct[k] for k in dct if not k.startswith('Not') and k != '_']
import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class MockEquality(FancyEqMixin):
    compareAttributes = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return f'MockEquality({self.name})'
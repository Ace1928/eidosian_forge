import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def _runCleanups(self, result):
    """
        Synchronously run any cleanups which have been added.
        """
    while len(self._cleanups) > 0:
        f, args, kwargs = self._cleanups.pop()
        try:
            f(*args, **kwargs)
        except BaseException:
            f = failure.Failure()
            result.addError(self, f)
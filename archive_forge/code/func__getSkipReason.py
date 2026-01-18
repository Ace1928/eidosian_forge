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
def _getSkipReason(self, method, skip):
    """
        Return the reason to use for skipping a test method.

        @param method: The method which produced the skip.
        @param skip: A L{unittest.SkipTest} instance raised by C{method}.
        """
    if len(skip.args) > 0:
        return skip.args[0]
    warnAboutFunction(method, 'Do not raise unittest.SkipTest with no arguments! Give a reason for skipping tests!')
    return skip
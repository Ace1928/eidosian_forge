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
def _getSuppress(self):
    """
        Returns any warning suppressions set for this test. Checks on the
        instance first, then the class, then the module, then packages. As
        soon as it finds something with a C{suppress} attribute, returns that.
        Returns any empty list (i.e. suppress no warnings) if it cannot find
        anything. See L{TestCase} docstring for more details.
        """
    return util.acquireAttribute(self._parents, 'suppress', [])
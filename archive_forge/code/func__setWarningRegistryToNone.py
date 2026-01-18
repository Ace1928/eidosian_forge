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
def _setWarningRegistryToNone(modules):
    """
    Disable the per-module cache for every module found in C{modules}, typically
    C{sys.modules}.

    @param modules: Dictionary of modules, typically sys.module dict
    """
    for v in list(modules.values()):
        if v is not None:
            try:
                v.__warningregistry__ = None
            except BaseException:
                pass
import doctest
import importlib
import inspect
import os
import sys
import types
import unittest as pyunit
import warnings
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from typing import Callable, Generator, List, Optional, TextIO, Type, Union
from zope.interface import implementer
from attrs import define
from typing_extensions import ParamSpec, Protocol, TypeAlias, TypeGuard
from twisted.internet import defer
from twisted.python import failure, filepath, log, modules, reflect
from twisted.trial import unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator, _iterateTests
from twisted.trial._synctest import _logObserver
from twisted.trial.itrial import ITestCase
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.unittest import TestSuite
from . import itrial
class DestructiveTestSuite(TestSuite):
    """
    A test suite which remove the tests once run, to minimize memory usage.
    """

    def run(self, result):
        """
        Almost the same as L{TestSuite.run}, but with C{self._tests} being
        empty at the end.
        """
        while self._tests:
            if result.shouldStop:
                break
            test = self._tests.pop(0)
            test(result)
        return result
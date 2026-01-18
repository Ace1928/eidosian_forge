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
class TrialSuite(TestSuite):
    """
    Suite to wrap around every single test in a C{trial} run. Used internally
    by Trial to set up things necessary for Trial tests to work, regardless of
    what context they are run in.
    """

    def __init__(self, tests=(), forceGarbageCollection=False):
        if forceGarbageCollection:
            newTests = []
            for test in tests:
                test = unittest.decorate(test, _ForceGarbageCollectionDecorator)
                newTests.append(test)
            tests = newTests
        suite = LoggedSuite(tests)
        super().__init__([suite])

    def _bail(self):
        from twisted.internet import reactor
        d = defer.Deferred()
        reactor.addSystemEventTrigger('after', 'shutdown', lambda: d.callback(None))
        reactor.fireSystemEvent('shutdown')
        unittest.TestCase('mktemp')._wait(d)

    def run(self, result):
        try:
            TestSuite.run(self, result)
        finally:
            self._bail()
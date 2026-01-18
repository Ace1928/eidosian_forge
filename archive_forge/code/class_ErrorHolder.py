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
class ErrorHolder(TestHolder):
    """
    Used to insert arbitrary errors into a test suite run. Provides enough
    methods to look like a C{TestCase}, however, when it is run, it simply adds
    an error to the C{TestResult}. The most common use-case is for when a
    module fails to import.
    """

    def __init__(self, description, error):
        """
        @param description: A string used by C{TestResult}s to identify this
        error. Generally, this is the name of a module that failed to import.

        @param error: The error to be added to the result. Can be an `exc_info`
        tuple or a L{twisted.python.failure.Failure}.
        """
        super().__init__(description)
        self.error = util.excInfoOrFailureToExcInfo(error)

    def __repr__(self) -> str:
        return '<ErrorHolder description={!r} error={!r}>'.format(self.description, self.error[1])

    def run(self, result):
        """
        Run the test, reporting the error.

        @param result: The C{TestResult} to store the results in.
        @type result: L{twisted.trial.itrial.IReporter}.
        """
        result.startTest(self)
        result.addError(self, self.error)
        result.stopTest(self)
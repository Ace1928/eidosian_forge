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
@contextmanager
def _logFile(logfile: str) -> Generator[None, None, None]:
    """
    A context manager which adds a log observer and then removes it.

    @param logfile: C{"-"} f or stdout logging, otherwise the path to a log
        file to which to write.
    """
    if logfile == '-':
        logFile = sys.stdout
    else:
        logFile = util.openTestLog(filepath.FilePath(logfile))
    logFileObserver = log.FileLogObserver(logFile)
    observerFunction = logFileObserver.emit
    log.startLoggingWithObserver(observerFunction, 0)
    yield
    log.removeObserver(observerFunction)
    logFile.close()
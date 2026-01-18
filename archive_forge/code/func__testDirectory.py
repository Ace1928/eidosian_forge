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
def _testDirectory(workingDirectory: str) -> Generator[None, None, None]:
    """
    A context manager which obtains a lock on a trial working directory
    and enters (L{os.chdir}) it and then reverses these things.

    @param workingDirectory: A pattern for the basename of the working
        directory to acquire.
    """
    currentDir = os.getcwd()
    base = filepath.FilePath(workingDirectory)
    testdir, testDirLock = util._unusedTestDirectory(base)
    os.chdir(testdir.path)
    yield
    os.chdir(currentDir)
    testDirLock.unlock()
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
def _uniqueTests(self, things):
    """
        Gather unique suite objects from loaded things. This will guarantee
        uniqueness of inherited methods on TestCases which would otherwise hash
        to same value and collapse to one test unexpectedly if using simpler
        means: e.g. set().
        """
    seen = set()
    for testthing in things:
        testthings = testthing._tests
        for thing in testthings:
            if str(thing) not in seen:
                yield thing
                seen.add(str(thing))
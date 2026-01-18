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
def _getMethodNameInClass(method):
    """
    Find the attribute name on the method's class which refers to the method.

    For some methods, notably decorators which have not had __name__ set correctly:

    getattr(method.im_class, method.__name__) != method
    """
    if getattr(method.im_class, method.__name__, object()) != method:
        for alias in dir(method.im_class):
            if getattr(method.im_class, alias, object()) == method:
                return alias
    return method.__name__
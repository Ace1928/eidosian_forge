from __future__ import annotations
import os
import sys
import unittest as pyunit
from hashlib import md5
from operator import attrgetter
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator
from hamcrest import assert_that, equal_to, has_properties
from hamcrest.core.matcher import Matcher
from twisted.python import filepath, util
from twisted.python.modules import PythonAttribute, PythonModule, getModule
from twisted.python.reflect import ModuleNotFound
from twisted.trial import reporter, runner, unittest
from twisted.trial._asyncrunner import _iterateTests
from twisted.trial.itrial import ITestCase
from twisted.trial.test import packages
from .matchers import after
class FinderPy3Tests(packages.SysPathManglingTest):

    def setUp(self) -> None:
        super().setUp()
        self.loader = runner.TestLoader()

    def test_findNonModule(self) -> None:
        """
        findByName, if given something findable up until the last entry, will
        raise AttributeError (as it cannot tell if 'nonexistent' here is
        supposed to be a module or a class).
        """
        self.assertRaises(AttributeError, self.loader.findByName, 'twisted.trial.test.nonexistent')

    def test_findNonPackage(self) -> None:
        self.assertRaises(ModuleNotFound, self.loader.findByName, 'nonextant')

    def test_findNonFile(self) -> None:
        """
        findByName, given a file path that doesn't exist, will raise a
        ValueError saying that it is not a Python file.
        """
        path = util.sibpath(__file__, 'nonexistent.py')
        self.assertRaises(ValueError, self.loader.findByName, path)

    def test_findFileWithImportError(self) -> None:
        """
        findByName will re-raise ImportErrors inside modules that it has found
        and imported.
        """
        self.assertRaises(ImportError, self.loader.findByName, 'unimportablepackage.test_module')
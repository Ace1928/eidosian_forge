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
class FileTests(packages.SysPathManglingTest):
    """
    Tests for L{runner.filenameToModule}.
    """

    def test_notFile(self) -> None:
        """
        L{runner.filenameToModule} raises a C{ValueError} when a non-existing
        file is passed.
        """
        err = self.assertRaises(ValueError, runner.filenameToModule, 'it')
        self.assertEqual(str(err), "'it' doesn't exist")

    def test_moduleInPath(self) -> None:
        """
        If the file in question is a module on the Python path, then it should
        properly import and return that module.
        """
        sample1 = runner.filenameToModule(util.sibpath(__file__, 'sample.py'))
        from twisted.trial.test import sample as sample2
        self.assertEqual(sample2, sample1)

    def test_moduleNotInPath(self) -> None:
        """
        If passed the path to a file containing the implementation of a
        module within a package which is not on the import path,
        L{runner.filenameToModule} returns a module object loosely
        resembling the module defined by that file anyway.
        """
        self.mangleSysPath(self.oldPath)
        sample1 = runner.filenameToModule(os.path.join(self.parent, 'goodpackage', 'test_sample.py'))
        self.assertEqual(sample1.__name__, 'goodpackage.test_sample')
        self.cleanUpModules()
        self.mangleSysPath(self.newPath)
        from goodpackage import test_sample as sample2
        self.assertIsNot(sample1, sample2)
        assert_that(sample1, looselyResembles(sample2))

    def test_packageInPath(self) -> None:
        """
        If the file in question is a package on the Python path, then it should
        properly import and return that package.
        """
        package1 = runner.filenameToModule(os.path.join(self.parent, 'goodpackage'))
        self.assertIs(package1, sys.modules['goodpackage'])

    def test_packageNotInPath(self) -> None:
        """
        If passed the path to a directory which represents a package which
        is not on the import path, L{runner.filenameToModule} returns a
        module object loosely resembling the package defined by that
        directory anyway.
        """
        self.mangleSysPath(self.oldPath)
        package1 = runner.filenameToModule(os.path.join(self.parent, 'goodpackage'))
        self.assertEqual(package1.__name__, 'goodpackage')
        self.cleanUpModules()
        self.mangleSysPath(self.newPath)
        import goodpackage
        self.assertIsNot(package1, goodpackage)
        assert_that(package1, looselyResembles(goodpackage))

    def test_directoryNotPackage(self) -> None:
        """
        L{runner.filenameToModule} raises a C{ValueError} when the name of an
        empty directory is passed that isn't considered a valid Python package
        because it doesn't contain a C{__init__.py} file.
        """
        emptyDir = filepath.FilePath(self.parent).child('emptyDirectory')
        emptyDir.createDirectory()
        err = self.assertRaises(ValueError, runner.filenameToModule, emptyDir.path)
        self.assertEqual(str(err), f'{emptyDir.path!r} is not a package directory')

    def test_filenameNotPython(self) -> None:
        """
        L{runner.filenameToModule} raises a C{SyntaxError} when a non-Python
        file is passed.
        """
        filename = filepath.FilePath(self.parent).child('notpython')
        filename.setContent(b"This isn't python")
        self.assertRaises(SyntaxError, runner.filenameToModule, filename.path)

    def test_filenameMatchesPackage(self) -> None:
        """
        The C{__file__} attribute of the module should match the package name.
        """
        filename = filepath.FilePath(self.parent).child('goodpackage.py')
        filename.setContent(packages.testModule.encode('utf8'))
        try:
            module = runner.filenameToModule(filename.path)
            self.assertEqual(filename.path, module.__file__)
        finally:
            filename.remove()

    def test_directory(self) -> None:
        """
        Test loader against a filesystem directory containing an empty
        C{__init__.py} file. It should handle 'path' and 'path/' the same way.
        """
        goodDir = filepath.FilePath(self.parent).child('goodDirectory')
        goodDir.createDirectory()
        goodDir.child('__init__.py').setContent(b'')
        try:
            module = runner.filenameToModule(goodDir.path)
            self.assertTrue(module.__name__.endswith('goodDirectory'))
            module = runner.filenameToModule(goodDir.path + os.path.sep)
            self.assertTrue(module.__name__.endswith('goodDirectory'))
        finally:
            goodDir.remove()
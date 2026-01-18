from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
class TestModuleTests(unittest.SynchronousTestCase):

    def setUp(self) -> None:
        self.config = trial.Options()

    def tearDown(self) -> None:
        self.config = None

    def test_testNames(self) -> None:
        """
        Check that the testNames helper method accurately collects the
        names of tests in suite.
        """
        self.assertEqual(testNames(self), [self.id()])

    def assertSuitesEqual(self, test1: TestSuite, names: list[str]) -> None:
        loader = TestLoader()
        names1 = testNames(test1)
        names2 = testNames(TestSuite(map(loader.loadByName, names)))
        names1.sort()
        names2.sort()
        self.assertEqual(names1, names2)

    def test_baseState(self) -> None:
        self.assertEqual(0, len(self.config['tests']))

    def test_testmoduleOnModule(self) -> None:
        """
        Check that --testmodule loads a suite which contains the tests
        referred to in test-case-name inside its parameter.
        """
        self.config.opt_testmodule(sibpath('moduletest.py'))
        self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.test_log'])

    def test_testmoduleTwice(self) -> None:
        """
        When the same module is specified with two --testmodule flags, it
        should only appear once in the suite.
        """
        self.config.opt_testmodule(sibpath('moduletest.py'))
        self.config.opt_testmodule(sibpath('moduletest.py'))
        self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.test_log'])

    def test_testmoduleOnSourceAndTarget(self) -> None:
        """
        If --testmodule is specified twice, once for module A and once for
        a module which refers to module A, then make sure module A is only
        added once.
        """
        self.config.opt_testmodule(sibpath('moduletest.py'))
        self.config.opt_testmodule(sibpath('test_log.py'))
        self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.test_log'])

    def test_testmoduleOnSelfModule(self) -> None:
        """
        When given a module that refers to *itself* in the test-case-name
        variable, check that --testmodule only adds the tests once.
        """
        self.config.opt_testmodule(sibpath('moduleself.py'))
        self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.moduleself'])

    def test_testmoduleOnScript(self) -> None:
        """
        Check that --testmodule loads tests referred to in test-case-name
        buffer variables.
        """
        self.config.opt_testmodule(sibpath('scripttest.py'))
        self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.test_log', 'twisted.trial.test.test_runner'])

    def test_testmoduleOnNonexistentFile(self) -> None:
        """
        Check that --testmodule displays a meaningful error message when
        passed a non-existent filename.
        """
        buffy = StringIO()
        stderr, sys.stderr = (sys.stderr, buffy)
        filename = 'test_thisbetternoteverexist.py'
        try:
            self.config.opt_testmodule(filename)
            self.assertEqual(0, len(self.config['tests']))
            self.assertEqual(f"File {filename!r} doesn't exist\n", buffy.getvalue())
        finally:
            sys.stderr = stderr

    def test_testmoduleOnEmptyVars(self) -> None:
        """
        Check that --testmodule adds no tests to the suite for modules
        which lack test-case-name buffer variables.
        """
        self.config.opt_testmodule(sibpath('novars.py'))
        self.assertEqual(0, len(self.config['tests']))

    def test_testmoduleOnModuleName(self) -> None:
        """
        Check that --testmodule does *not* support module names as arguments
        and that it displays a meaningful error message.
        """
        buffy = StringIO()
        stderr, sys.stderr = (sys.stderr, buffy)
        moduleName = 'twisted.trial.test.test_script'
        try:
            self.config.opt_testmodule(moduleName)
            self.assertEqual(0, len(self.config['tests']))
            self.assertEqual(f"File {moduleName!r} doesn't exist\n", buffy.getvalue())
        finally:
            sys.stderr = stderr

    def test_parseLocalVariable(self) -> None:
        declaration = '-*- test-case-name: twisted.trial.test.test_tests -*-'
        localVars = trial._parseLocalVariables(declaration)
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_tests'}, localVars)

    def test_trailingSemicolon(self) -> None:
        declaration = '-*- test-case-name: twisted.trial.test.test_tests; -*-'
        localVars = trial._parseLocalVariables(declaration)
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_tests'}, localVars)

    def test_parseLocalVariables(self) -> None:
        declaration = '-*- test-case-name: twisted.trial.test.test_tests; foo: bar -*-'
        localVars = trial._parseLocalVariables(declaration)
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_tests', 'foo': 'bar'}, localVars)

    def test_surroundingGuff(self) -> None:
        declaration = '## -*- test-case-name: twisted.trial.test.test_tests -*- #'
        localVars = trial._parseLocalVariables(declaration)
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_tests'}, localVars)

    def test_invalidLine(self) -> None:
        self.assertRaises(ValueError, trial._parseLocalVariables, 'foo')

    def test_invalidDeclaration(self) -> None:
        self.assertRaises(ValueError, trial._parseLocalVariables, '-*- foo -*-')
        self.assertRaises(ValueError, trial._parseLocalVariables, '-*- foo: bar; qux -*-')
        self.assertRaises(ValueError, trial._parseLocalVariables, '-*- foo: bar: baz; qux: qax -*-')

    def test_variablesFromFile(self) -> None:
        localVars = trial.loadLocalVariables(sibpath('moduletest.py'))
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_log'}, localVars)

    def test_noVariablesInFile(self) -> None:
        localVars = trial.loadLocalVariables(sibpath('novars.py'))
        self.assertEqual({}, localVars)

    def test_variablesFromScript(self) -> None:
        localVars = trial.loadLocalVariables(sibpath('scripttest.py'))
        self.assertEqual({'test-case-name': 'twisted.trial.test.test_log,twisted.trial.test.test_runner'}, localVars)

    def test_getTestModules(self) -> None:
        modules = trial.getTestModules(sibpath('moduletest.py'))
        self.assertEqual(modules, ['twisted.trial.test.test_log'])

    def test_getTestModules_noVars(self) -> None:
        modules = trial.getTestModules(sibpath('novars.py'))
        self.assertEqual(len(modules), 0)

    def test_getTestModules_multiple(self) -> None:
        modules = trial.getTestModules(sibpath('scripttest.py'))
        self.assertEqual(set(modules), {'twisted.trial.test.test_log', 'twisted.trial.test.test_runner'})

    def test_looksLikeTestModule(self) -> None:
        for filename in ['test_script.py', 'twisted/trial/test/test_script.py']:
            self.assertTrue(trial.isTestFile(filename), f'{filename!r} should be a test file')
        for filename in ['twisted/trial/test/moduletest.py', sibpath('scripttest.py'), sibpath('test_foo.bat')]:
            self.assertFalse(trial.isTestFile(filename), f'{filename!r} should *not* be a test file')
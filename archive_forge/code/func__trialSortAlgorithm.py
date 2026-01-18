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
def _trialSortAlgorithm(self, sorter: Callable[[PythonModule | PythonAttribute], SupportsRichComparison]) -> Generator[PythonModule | PythonAttribute, None, None]:
    """
        Right now, halfway by accident, trial sorts like this:

            1. all modules are grouped together in one list and sorted.

            2. within each module, the classes are grouped together in one list
               and sorted.

            3. finally within each class, each test method is grouped together
               in a list and sorted.

        This attempts to return a sorted list of testable thingies following
        those rules, so that we can compare the behavior of loadPackage.

        The things that show as 'cases' are errors from modules which failed to
        import, and test methods.  Let's gather all those together.
        """
    pkg = getModule('uberpackage')
    testModules = []
    for testModule in pkg.walkModules():
        if testModule.name.split('.')[-1].startswith('test_'):
            testModules.append(testModule)
    sortedModules = sorted(testModules, key=sorter)
    for modinfo in sortedModules:
        module = modinfo.load(None)
        if module is None:
            yield modinfo
        else:
            testClasses = []
            for attrib in modinfo.iterAttributes():
                if runner.isTestCase(attrib.load()):
                    testClasses.append(attrib)
            sortedClasses = sorted(testClasses, key=sorter)
            for clsinfo in sortedClasses:
                testMethods = []
                for attr in clsinfo.iterAttributes():
                    if attr.name.split('.')[-1].startswith('test'):
                        testMethods.append(attr)
                sortedMethods = sorted(testMethods, key=sorter)
                yield from sortedMethods
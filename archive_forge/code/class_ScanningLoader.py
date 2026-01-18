import os
import operator
import sys
import contextlib
import itertools
import unittest
from distutils.errors import DistutilsError, DistutilsOptionError
from distutils import log
from unittest import TestLoader
from pkg_resources import (
from .._importlib import metadata
from setuptools import Command
from setuptools.extern.more_itertools import unique_everseen
from setuptools.extern.jaraco.functools import pass_none
class ScanningLoader(TestLoader):

    def __init__(self):
        TestLoader.__init__(self)
        self._visited = set()

    def loadTestsFromModule(self, module, pattern=None):
        """Return a suite of all tests cases contained in the given module

        If the module is a package, load tests from all the modules in it.
        If the module has an ``additional_tests`` function, call it and add
        the return value to the tests.
        """
        if module in self._visited:
            return None
        self._visited.add(module)
        tests = []
        tests.append(TestLoader.loadTestsFromModule(self, module))
        if hasattr(module, 'additional_tests'):
            tests.append(module.additional_tests())
        if hasattr(module, '__path__'):
            for file in resource_listdir(module.__name__, ''):
                if file.endswith('.py') and file != '__init__.py':
                    submodule = module.__name__ + '.' + file[:-3]
                elif resource_exists(module.__name__, file + '/__init__.py'):
                    submodule = module.__name__ + '.' + file
                else:
                    continue
                tests.append(self.loadTestsFromName(submodule))
        if len(tests) != 1:
            return self.suiteClass(tests)
        else:
            return tests[0]
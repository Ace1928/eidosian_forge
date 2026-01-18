import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class FilteredByModuleTestLoader(TestLoader):
    """A test loader that import only the needed modules."""

    def __init__(self, needs_module):
        """Constructor.

        :param needs_module: a callable taking a module name as a
            parameter returing True if the module should be loaded.
        """
        TestLoader.__init__(self)
        self.needs_module = needs_module

    def loadTestsFromModuleName(self, name):
        if self.needs_module(name):
            return TestLoader.loadTestsFromModuleName(self, name)
        else:
            return self.suiteClass()
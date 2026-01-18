import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
class Py2Fixer(object):
    """
    An import hook class that uses lib2to3 for source-to-source translation of
    Py2 code to Py3.
    """
    PY2FIXER = True

    def __init__(self):
        self.found = None
        self.base_exclude_paths = ['future', 'past']
        self.exclude_paths = copy.copy(self.base_exclude_paths)
        self.include_paths = []

    def include(self, paths):
        """
        Pass in a sequence of module names such as 'plotrique.plotting' that,
        if present at the leftmost side of the full package name, would
        specify the module to be transformed from Py2 to Py3.
        """
        self.include_paths += paths

    def exclude(self, paths):
        """
        Pass in a sequence of strings such as 'mymodule' that, if
        present at the leftmost side of the full package name, would cause
        the module not to undergo any source transformation.
        """
        self.exclude_paths += paths

    def find_module(self, fullname, path=None):
        logger.debug('Running find_module: (%s, %s)', fullname, path)
        loader = PathFinder.find_module(fullname, path)
        if not loader:
            logger.debug('Py2Fixer could not find %s', fullname)
            return None
        loader.__class__ = PastSourceFileLoader
        loader.exclude_paths = self.exclude_paths
        loader.include_paths = self.include_paths
        return loader

    def find_spec(self, fullname, path=None, target=None):
        logger.debug('Running find_spec: (%s, %s, %s)', fullname, path, target)
        spec = PathFinder.find_spec(fullname, path, target)
        if not spec:
            logger.debug('Py2Fixer could not find %s', fullname)
            return None
        spec.loader.__class__ = PastSourceFileLoader
        spec.loader.exclude_paths = self.exclude_paths
        spec.loader.include_paths = self.include_paths
        return spec
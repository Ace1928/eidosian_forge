import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from jedi.inference.compiled import subprocess  # noqa: E402
class _ExactImporter(MetaPathFinder):

    def __init__(self, path_dct):
        self._path_dct = path_dct

    def find_spec(self, fullname, path=None, target=None):
        if path is None and fullname in self._path_dct:
            p = self._path_dct[fullname]
            spec = PathFinder.find_spec(fullname, path=[p], target=target)
            return spec
        return None
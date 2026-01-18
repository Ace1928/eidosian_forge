import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
class ShimImporter(importlib.abc.MetaPathFinder):
    """Import hook for a shim.

    This ensures that submodule imports return the real target module,
    not a clone that will confuse `is` and `isinstance` checks.
    """

    def __init__(self, src, mirror):
        self.src = src
        self.mirror = mirror

    def _mirror_name(self, fullname):
        """get the name of the mirrored module"""
        return self.mirror + fullname[len(self.src):]

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.src + '.'):
            mirror_name = self._mirror_name(fullname)
            return importlib.util.find_spec(mirror_name)
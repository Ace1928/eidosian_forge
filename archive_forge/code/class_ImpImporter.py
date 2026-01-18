from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings
class ImpImporter:
    """PEP 302 Finder that wraps Python's "classic" import algorithm

    ImpImporter(dirname) produces a PEP 302 finder that searches that
    directory.  ImpImporter(None) produces a PEP 302 finder that searches
    the current sys.path, plus any modules that are frozen or built-in.

    Note that ImpImporter does not currently support being used by placement
    on sys.meta_path.
    """

    def __init__(self, path=None):
        global imp
        warnings.warn("This emulation is deprecated and slated for removal in Python 3.12; use 'importlib' instead", DeprecationWarning)
        _import_imp()
        self.path = path

    def find_module(self, fullname, path=None):
        subname = fullname.split('.')[-1]
        if subname != fullname and self.path is None:
            return None
        if self.path is None:
            path = None
        else:
            path = [os.path.realpath(self.path)]
        try:
            file, filename, etc = imp.find_module(subname, path)
        except ImportError:
            return None
        return ImpLoader(fullname, file, filename, etc)

    def iter_modules(self, prefix=''):
        if self.path is None or not os.path.isdir(self.path):
            return
        yielded = {}
        import inspect
        try:
            filenames = os.listdir(self.path)
        except OSError:
            filenames = []
        filenames.sort()
        for fn in filenames:
            modname = inspect.getmodulename(fn)
            if modname == '__init__' or modname in yielded:
                continue
            path = os.path.join(self.path, fn)
            ispkg = False
            if not modname and os.path.isdir(path) and ('.' not in fn):
                modname = fn
                try:
                    dircontents = os.listdir(path)
                except OSError:
                    dircontents = []
                for fn in dircontents:
                    subname = inspect.getmodulename(fn)
                    if subname == '__init__':
                        ispkg = True
                        break
                else:
                    continue
            if modname and '.' not in modname:
                yielded[modname] = 1
                yield (prefix + modname, ispkg)
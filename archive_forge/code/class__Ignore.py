import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
class _Ignore:

    def __init__(self, modules=None, dirs=None):
        self._mods = set() if not modules else set(modules)
        self._dirs = [] if not dirs else [os.path.normpath(d) for d in dirs]
        self._ignore = {'<string>': 1}

    def names(self, filename, modulename):
        if modulename in self._ignore:
            return self._ignore[modulename]
        if modulename in self._mods:
            self._ignore[modulename] = 1
            return 1
        for mod in self._mods:
            if modulename.startswith(mod + '.'):
                self._ignore[modulename] = 1
                return 1
        if filename is None:
            self._ignore[modulename] = 1
            return 1
        for d in self._dirs:
            if filename.startswith(d + os.sep):
                self._ignore[modulename] = 1
                return 1
        self._ignore[modulename] = 0
        return 0
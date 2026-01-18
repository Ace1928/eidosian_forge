from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
import pickle
import sys
import tempfile
import uuid
import warnings
from numba.misc.appdirs import AppDirs
import numba
from numba.core.errors import NumbaWarning
from numba.core.base import BaseContext
from numba.core.codegen import CodeLibrary
from numba.core.compiler import CompileResult
from numba.core import config, compiler
from numba.core.serialize import dumps
class _UserProvidedCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator that always point to the user provided directory in
    `numba.config.CACHE_DIR`
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        cache_subpath = self.get_suitable_cache_subpath(py_file)
        self._cache_path = os.path.join(config.CACHE_DIR, cache_subpath)

    def get_cache_path(self):
        return self._cache_path

    @classmethod
    def from_function(cls, py_func, py_file):
        if not config.CACHE_DIR:
            return
        parent = super(_UserProvidedCacheLocator, cls)
        return parent.from_function(py_func, py_file)
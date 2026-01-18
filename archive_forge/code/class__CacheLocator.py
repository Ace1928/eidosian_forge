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
class _CacheLocator(metaclass=ABCMeta):
    """
    A filesystem locator for caching a given function.
    """

    def ensure_cache_path(self):
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()

    @abstractmethod
    def get_cache_path(self):
        """
        Return the directory the function is cached in.
        """

    @abstractmethod
    def get_source_stamp(self):
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """

    @abstractmethod
    def get_disambiguator(self):
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """

    @classmethod
    def from_function(cls, py_func, py_file):
        """
        Create a locator instance for the given function located in the
        given file.
        """
        raise NotImplementedError

    @classmethod
    def get_suitable_cache_subpath(cls, py_file):
        """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """
        path = os.path.abspath(py_file)
        subpath = os.path.dirname(path)
        parentdir = os.path.split(subpath)[-1]
        hashed = hashlib.sha1(subpath.encode()).hexdigest()
        return '_'.join([parentdir, hashed])
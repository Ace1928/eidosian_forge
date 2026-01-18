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
def _load_index(self):
    """
        Load the cache index and return it as a dictionary (possibly
        empty if cache is empty or obsolete).
        """
    try:
        with open(self._index_path, 'rb') as f:
            version = pickle.load(f)
            data = f.read()
    except FileNotFoundError:
        return {}
    if version != self._version:
        return {}
    stamp, overloads = pickle.loads(data)
    _cache_log('[cache] index loaded from %r', self._index_path)
    if stamp != self._source_stamp:
        return {}
    else:
        return overloads
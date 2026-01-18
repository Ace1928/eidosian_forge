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
class CompileResultCacheImpl(CacheImpl):
    """
    Implements the logic to cache CompileResult objects.
    """

    def reduce(self, cres):
        """
        Returns a serialized CompileResult
        """
        return cres._reduce()

    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CompileResult
        """
        return compiler.CompileResult._rebuild(target_context, *payload)

    def check_cachable(self, cres):
        """
        Check cachability of the given compile result.
        """
        cannot_cache = None
        if any((not x.can_cache for x in cres.lifted)):
            cannot_cache = 'as it uses lifted code'
        elif cres.library.has_dynamic_globals:
            cannot_cache = 'as it uses dynamic globals (such as ctypes pointers and large global arrays)'
        if cannot_cache:
            msg = 'Cannot cache compiled function "%s" %s' % (cres.fndesc.qualname.split('.')[-1], cannot_cache)
            warnings.warn_explicit(msg, NumbaWarning, self._locator._py_file, self._lineno)
            return False
        return True
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
class CodeLibraryCacheImpl(CacheImpl):
    """
    Implements the logic to cache CodeLibrary objects.
    """
    _filename_prefix = None

    def reduce(self, codelib):
        """
        Returns a serialized CodeLibrary
        """
        return codelib.serialize_using_object_code()

    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CodeLibrary
        """
        return target_context.codegen().unserialize_library(payload)

    def check_cachable(self, codelib):
        """
        Check cachability of the given CodeLibrary.
        """
        return not codelib.has_dynamic_globals

    def get_filename_base(self, fullname, abiflags):
        parent = super(CodeLibraryCacheImpl, self)
        res = parent.get_filename_base(fullname, abiflags)
        return '-'.join([self._filename_prefix, res])
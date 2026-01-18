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
def _load_overload(self, sig, target_context):
    if not self._enabled:
        return
    key = self._index_key(sig, target_context.codegen())
    data = self._cache_file.load(key)
    if data is not None:
        data = self._impl.rebuild(target_context, data)
    return data
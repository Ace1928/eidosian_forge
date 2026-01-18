from __future__ import with_statement
import logging
import os
from textwrap import dedent
import time
import pathlib
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import weakref
from datetime import timedelta
from tokenize import open as open_py_source
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from .logger import Logger, format_time, pformat
from ._store_backends import StoreBackendBase, FileSystemStoreBackend
from ._store_backends import CacheWarning  # noqa
class MemorizedResult(Logger):
    """Object representing a cached value.

    Attributes
    ----------
    location: str
        The location of joblib cache. Depends on the store backend used.

    func: function or str
        function whose output is cached. The string case is intended only for
        instantiation based on the output of repr() on another instance.
        (namely eval(repr(memorized_instance)) works).

    argument_hash: str
        hash of the function arguments.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local'.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache numpy arrays. See
        numpy.load for the meaning of the different values.

    verbose: int
        verbosity level (0 means no message).

    timestamp, metadata: string
        for internal use only.
    """

    def __init__(self, location, func, args_id, backend='local', mmap_mode=None, verbose=0, timestamp=None, metadata=None):
        Logger.__init__(self)
        self.func_id = _build_func_identifier(func)
        if isinstance(func, str):
            self.func = func
        else:
            self.func = self.func_id
        self.args_id = args_id
        self.store_backend = _store_backend_factory(backend, location, verbose=verbose)
        self.mmap_mode = mmap_mode
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = self.store_backend.get_metadata([self.func_id, self.args_id])
        self.duration = self.metadata.get('duration', None)
        self.verbose = verbose
        self.timestamp = timestamp

    @property
    def argument_hash(self):
        warnings.warn("The 'argument_hash' attribute has been deprecated in version 0.12 and will be removed in version 0.14.\nUse `args_id` attribute instead.", DeprecationWarning, stacklevel=2)
        return self.args_id

    def get(self):
        """Read value from cache and return it."""
        if self.verbose:
            msg = _format_load_msg(self.func_id, self.args_id, timestamp=self.timestamp, metadata=self.metadata)
        else:
            msg = None
        try:
            return self.store_backend.load_item([self.func_id, self.args_id], msg=msg, verbose=self.verbose)
        except ValueError as exc:
            new_exc = KeyError("Error while trying to load a MemorizedResult's value. It seems that this folder is corrupted : {}".format(os.path.join(self.store_backend.location, self.func_id, self.args_id)))
            raise new_exc from exc

    def clear(self):
        """Clear value from cache"""
        self.store_backend.clear_item([self.func_id, self.args_id])

    def __repr__(self):
        return '{class_name}(location="{location}", func="{func}", args_id="{args_id}")'.format(class_name=self.__class__.__name__, location=self.store_backend.location, func=self.func, args_id=self.args_id)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state
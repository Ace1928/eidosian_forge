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
def _cached_call(self, args, kwargs, shelving=False):
    """Call wrapped function and cache result, or read cache if available.

        This function returns the wrapped function output and some metadata.

        Arguments:
        ----------

        args, kwargs: list and dict
            input arguments for wrapped function

        shelving: bool
            True when called via the call_and_shelve function.


        Returns
        -------
        output: value or tuple or None
            Output of the wrapped function.
            If shelving is True and the call has been already cached,
            output is None.

        argument_hash: string
            Hash of function arguments.

        metadata: dict
            Some metadata about wrapped function call (see _persist_input()).
        """
    func_id, args_id = self._get_output_identifiers(*args, **kwargs)
    metadata = None
    msg = None
    must_call = False
    if self._verbose >= 20:
        logging.basicConfig(level=logging.INFO)
        _, name = get_func_name(self.func)
        location = self.store_backend.get_cached_func_info([func_id])['location']
        _, signature = format_signature(self.func, *args, **kwargs)
        self.info(dedent(f'\n                        Querying {name} with signature\n                        {signature}.\n\n                        (argument hash {args_id})\n\n                        The store location is {location}.\n                        '))
    if self._is_in_cache_and_valid([func_id, args_id]):
        try:
            t0 = time.time()
            if self._verbose:
                msg = _format_load_msg(func_id, args_id, timestamp=self.timestamp, metadata=metadata)
            if not shelving:
                out = self.store_backend.load_item([func_id, args_id], msg=msg, verbose=self._verbose)
            else:
                out = None
            if self._verbose > 4:
                t = time.time() - t0
                _, name = get_func_name(self.func)
                msg = '%s cache loaded - %s' % (name, format_time(t))
                print(max(0, 80 - len(msg)) * '_' + msg)
        except Exception:
            _, signature = format_signature(self.func, *args, **kwargs)
            self.warn('Exception while loading results for {}\n {}'.format(signature, traceback.format_exc()))
            must_call = True
    else:
        if self._verbose > 10:
            _, name = get_func_name(self.func)
            self.warn('Computing func {0}, argument hash {1} in location {2}'.format(name, args_id, self.store_backend.get_cached_func_info([func_id])['location']))
        must_call = True
    if must_call:
        out, metadata = self.call(*args, **kwargs)
        if self.mmap_mode is not None:
            if self._verbose:
                msg = _format_load_msg(func_id, args_id, timestamp=self.timestamp, metadata=metadata)
            out = self.store_backend.load_item([func_id, args_id], msg=msg, verbose=self._verbose)
    return (out, args_id, metadata)
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
def _persist_input(self, duration, args, kwargs, this_duration_limit=0.5):
    """ Save a small summary of the call using json format in the
            output directory.

            output_dir: string
                directory where to write metadata.

            duration: float
                time taken by hashing input arguments, calling the wrapped
                function and persisting its output.

            args, kwargs: list and dict
                input arguments for wrapped function

            this_duration_limit: float
                Max execution time for this function before issuing a warning.
        """
    start_time = time.time()
    argument_dict = filter_args(self.func, self.ignore, args, kwargs)
    input_repr = dict(((k, repr(v)) for k, v in argument_dict.items()))
    metadata = {'duration': duration, 'input_args': input_repr, 'time': start_time}
    func_id, args_id = self._get_output_identifiers(*args, **kwargs)
    self.store_backend.store_metadata([func_id, args_id], metadata)
    this_duration = time.time() - start_time
    if this_duration > this_duration_limit:
        warnings.warn('Persisting input arguments took %.2fs to run.If this happens often in your code, it can cause performance problems (results will be correct in all cases). The reason for this is probably some large input arguments for a wrapped function.' % this_duration, stacklevel=5)
    return metadata
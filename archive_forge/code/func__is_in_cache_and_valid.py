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
def _is_in_cache_and_valid(self, path):
    """Check if the function call is cached and valid for given arguments.

        - Compare the function code with the one from the cached function,
        asserting if it has changed.
        - Check if the function call is present in the cache.
        - Call `cache_validation_callback` for user define cache validation.

        Returns True if the function call is in cache and can be used, and
        returns False otherwise.
        """
    if not self._check_previous_func_code(stacklevel=4):
        return False
    if not self.store_backend.contains_item(path):
        return False
    metadata = self.store_backend.get_metadata(path)
    if self.cache_validation_callback is not None and (not self.cache_validation_callback(metadata)):
        self.store_backend.clear_item(path)
        return False
    return True
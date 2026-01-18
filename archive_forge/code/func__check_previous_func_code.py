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
def _check_previous_func_code(self, stacklevel=2):
    """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
    try:
        if self.func in _FUNCTION_HASHES:
            func_hash = self._hash_func()
            if func_hash == _FUNCTION_HASHES[self.func]:
                return True
    except TypeError:
        pass
    func_code, source_file, first_line = self.func_code_info
    func_id = _build_func_identifier(self.func)
    try:
        old_func_code, old_first_line = extract_first_line(self.store_backend.get_cached_func_code([func_id]))
    except (IOError, OSError):
        self._write_func_code(func_code, first_line)
        return False
    if old_func_code == func_code:
        return True
    _, func_name = get_func_name(self.func, resolv_alias=False, win_characters=False)
    if old_first_line == first_line == -1 or func_name == '<lambda>':
        if not first_line == -1:
            func_description = '{0} ({1}:{2})'.format(func_name, source_file, first_line)
        else:
            func_description = func_name
        warnings.warn(JobLibCollisionWarning("Cannot detect name collisions for function '{0}'".format(func_description)), stacklevel=stacklevel)
    if not old_first_line == first_line and source_file is not None:
        possible_collision = False
        if os.path.exists(source_file):
            _, func_name = get_func_name(self.func, resolv_alias=False)
            num_lines = len(func_code.split('\n'))
            with open_py_source(source_file) as f:
                on_disk_func_code = f.readlines()[old_first_line - 1:old_first_line - 1 + num_lines - 1]
            on_disk_func_code = ''.join(on_disk_func_code)
            possible_collision = on_disk_func_code.rstrip() == old_func_code.rstrip()
        else:
            possible_collision = source_file.startswith('<doctest ')
        if possible_collision:
            warnings.warn(JobLibCollisionWarning("Possible name collisions between functions '%s' (%s:%i) and '%s' (%s:%i)" % (func_name, source_file, old_first_line, func_name, source_file, first_line)), stacklevel=stacklevel)
    if self._verbose > 10:
        _, func_name = get_func_name(self.func, resolv_alias=False)
        self.warn('Function {0} (identified by {1}) has changed.'.format(func_name, func_id))
    self.clear(warn=True)
    return False
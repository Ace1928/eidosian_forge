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
def _format_load_msg(func_id, args_id, timestamp=None, metadata=None):
    """ Helper function to format the message when loading the results.
    """
    signature = ''
    try:
        if metadata is not None:
            args = ', '.join(['%s=%s' % (name, value) for name, value in metadata['input_args'].items()])
            signature = '%s(%s)' % (os.path.basename(func_id), args)
        else:
            signature = os.path.basename(func_id)
    except KeyError:
        pass
    if timestamp is not None:
        ts_string = '{0: <16}'.format(format_time(time.time() - timestamp))
    else:
        ts_string = ''
    return '[Memory]{0}: Loading {1}'.format(ts_string, str(signature))
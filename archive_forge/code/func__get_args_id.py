import asyncio
import datetime
import functools
import inspect
import logging
import os
import pathlib
import pydoc
import re
import textwrap
import time
import tokenize
import traceback
import warnings
import weakref
from . import hashing
from ._store_backends import CacheWarning  # noqa
from ._store_backends import FileSystemStoreBackend, StoreBackendBase
from .func_inspect import (filter_args, format_call, format_signature,
from .logger import Logger, format_time, pformat
def _get_args_id(self, *args, **kwargs):
    """Return the input parameter hash of a result."""
    return hashing.hash(filter_args(self.func, self.ignore, args, kwargs), coerce_mmap=self.mmap_mode is not None)
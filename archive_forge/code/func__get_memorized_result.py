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
def _get_memorized_result(self, call_id, metadata=None):
    return MemorizedResult(self.store_backend, call_id, metadata=metadata, timestamp=self.timestamp, verbose=self._verbose - 1)
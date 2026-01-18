from __future__ import annotations
import os
import io
import abc
import zlib
import errno
import time
import struct
import sqlite3
import threading
import pickletools
import asyncio
import inspect
import dill as pkl
import functools as ft
import contextlib as cl
import warnings
from fileio.lib.types import File, FileLike
from typing import Any, Optional, Type, Dict, Union, Tuple, TYPE_CHECKING
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.constants import (
from lazyops.libs.sqlcache.config import SqlCacheConfig
from lazyops.libs.sqlcache.exceptions import (
from lazyops.libs.sqlcache.utils import (
def _execute_with_retry(statement, *args, **kwargs):
    start = time.time()
    while True:
        try:
            return sql(statement, *args, **kwargs)
        except sqlite3.OperationalError as exc:
            if str(exc) != 'database is locked':
                raise
            diff = time.time() - start
            if diff > 60:
                raise
            time.sleep(0.001)
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
def _row_insert(self, key, raw, now, columns):
    sql = self._sql
    expire_time, tag, size, mode, filename, value = columns
    sql(f'INSERT INTO {self.table_name}( key, raw, store_time, expire_time, access_time, access_count, tag, size, mode, filename, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (key, raw, now, expire_time, now, 0, tag, size, mode, filename, value))
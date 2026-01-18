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
def _select_delete(self, select, args, row_index: int=0, arg_index: int=0, retry: bool=False):
    count = 0
    delete = f'DELETE FROM {self.table_name} WHERE rowid IN (%s)'
    try:
        while True:
            with self._transact(retry) as (sql, cleanup):
                rows = sql(select, args).fetchall()
                if not rows:
                    break
                count += len(rows)
                sql(delete % ','.join((str(row[0]) for row in rows)))
                for row in rows:
                    args[arg_index] = row[row_index]
                    cleanup(row[-1])
    except SqlTimeout:
        raise SqlTimeout(count) from None
    return count
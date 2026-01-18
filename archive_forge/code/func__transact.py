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
@cl.contextmanager
def _transact(self, retry: bool=False, filename: str=None):
    sql = self._sql
    filenames = []
    tid = threading.get_ident()
    txn_id = self._txn_id
    if tid == txn_id:
        begin = False
    else:
        while True:
            try:
                sql('BEGIN IMMEDIATE')
                begin = True
                self._txn_id = tid
                break
            except sqlite3.OperationalError:
                if retry:
                    continue
                if filename is not None:
                    self.medium.remove(filename)
                raise SqlTimeout from None
    try:
        yield (sql, filenames.append)
    except BaseException:
        if begin:
            assert self._txn_id == tid
            self._txn_id = None
            sql('ROLLBACK')
        raise
    else:
        if begin:
            assert self._txn_id == tid
            self._txn_id = None
            sql('COMMIT')
        for name in filenames:
            if name is not None:
                self.medium.remove(name)
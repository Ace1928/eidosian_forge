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
@property
def _asql_retry(self):
    asql = self._asql

    async def _aexecute_with_retry(statement, *args, **kwargs):
        start = asyncio.get_event_loop().time()
        while True:
            try:
                return await asql(statement, *args, **kwargs)
            except sqlite3.OperationalError as exc:
                if str(exc) != 'database is locked':
                    raise
                diff = asyncio.get_event_loop().time() - start
                if diff > 60:
                    raise
                await asyncio.sleep(0.001)
    return _aexecute_with_retry
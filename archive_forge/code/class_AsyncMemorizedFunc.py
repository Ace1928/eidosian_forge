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
class AsyncMemorizedFunc(MemorizedFunc):

    async def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        return await out if asyncio.iscoroutine(out) else out

    async def call_and_shelve(self, *args, **kwargs):
        out = super().call_and_shelve(*args, **kwargs)
        return await out if asyncio.iscoroutine(out) else out

    async def call(self, *args, **kwargs):
        out = super().call(*args, **kwargs)
        return await out if asyncio.iscoroutine(out) else out

    async def _call(self, call_id, args, kwargs, shelving=False):
        self._before_call(args, kwargs)
        start_time = time.time()
        output = await self.func(*args, **kwargs)
        return self._after_call(call_id, args, kwargs, shelving, output, start_time)
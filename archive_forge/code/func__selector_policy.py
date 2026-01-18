import asyncio
import asyncio.events
import functools
import inspect
import io
import numbers
import os
import re
import threading
from contextlib import contextmanager
from glob import has_magic
from typing import TYPE_CHECKING, Iterable
from .callbacks import DEFAULT_CALLBACK
from .exceptions import FSTimeoutError
from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
from .spec import AbstractBufferedFile, AbstractFileSystem
from .utils import glob_translate, is_exception, other_paths
@contextmanager
def _selector_policy():
    original_policy = asyncio.get_event_loop_policy()
    try:
        if os.name == 'nt' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        yield
    finally:
        asyncio.set_event_loop_policy(original_policy)
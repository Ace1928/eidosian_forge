from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
def _make_function_key(cache_type: CacheType, func: types.FunctionType) -> str:
    """Create the unique key for a function's cache.

    A function's key is stable across reruns of the app, and changes when
    the function's source code changes.
    """
    func_hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    update_hash((func.__module__, func.__qualname__), hasher=func_hasher, cache_type=cache_type, hash_source=func)
    source_code: str | bytes
    try:
        source_code = inspect.getsource(func)
    except OSError as e:
        _LOGGER.debug("Failed to retrieve function's source code when building its key; falling back to bytecode. err={0}", e)
        source_code = func.__code__.co_code
    update_hash(source_code, hasher=func_hasher, cache_type=cache_type, hash_source=func)
    cache_key = func_hasher.hexdigest()
    return cache_key
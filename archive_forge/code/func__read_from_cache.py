from __future__ import annotations
import contextlib
import functools
import hashlib
import inspect
import math
import os
import pickle
import shutil
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterator, TypeVar, cast, overload
from cachetools import TTLCache
import streamlit as st
from streamlit import config, file_util, util
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.elements.spinner import spinner
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.errors import StreamlitAPIWarning
from streamlit.logger import get_logger
from streamlit.runtime.caching import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
from streamlit.runtime.legacy_caching.hashing import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.util import HASHLIB_KWARGS
def _read_from_cache(mem_cache: MemCache, key: str, persist: bool, allow_output_mutation: bool, func_or_code: Callable[..., Any], hash_funcs: HashFuncsDict | None=None) -> Any:
    """Read a value from the cache.

    Our goal is to read from memory if possible. If the data was mutated (hash
    changed), we show a warning. If reading from memory fails, we either read
    from disk or rerun the code.
    """
    try:
        return _read_from_mem_cache(mem_cache, key, allow_output_mutation, func_or_code, hash_funcs)
    except CachedObjectMutationError as e:
        handle_uncaught_app_exception(CachedObjectMutationWarning(e))
        return e.cached_value
    except CacheKeyNotFoundError as e:
        if persist:
            value = _read_from_disk_cache(key)
            _write_to_mem_cache(mem_cache, key, value, allow_output_mutation, func_or_code, hash_funcs)
            return value
        raise e
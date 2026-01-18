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
def _read_from_disk_cache(key: str) -> Any:
    path = file_util.get_streamlit_file_path('cache', '%s.pickle' % key)
    try:
        with file_util.streamlit_read(path, binary=True) as input:
            entry = pickle.load(input)
            value = entry.value
            _LOGGER.debug('Disk cache HIT: %s', type(value))
    except util.Error as e:
        _LOGGER.error(e)
        raise CacheError('Unable to read from cache: %s' % e)
    except FileNotFoundError:
        raise CacheKeyNotFoundError('Key not found in disk cache')
    return value
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
def _make_deprecation_warning(cached_value: Any) -> str:
    """Build a deprecation warning string for a cache function that has returned the given
    value.
    """
    typename = type(cached_value).__qualname__
    cache_type_rec = NEW_CACHE_FUNC_RECOMMENDATIONS.get(typename)
    if cache_type_rec is not None:
        return f"`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n`st.cache_data` or `st.cache_resource`. Based on this function's return value\nof type `{typename}`, we recommend using `st.{get_decorator_api_name(cache_type_rec)}`.\n\nMore information [in our docs]({CACHE_DOCS_URL})."
    return f"`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n`st.cache_data` or `st.cache_resource`.\n\nMore information [in our docs]({CACHE_DOCS_URL})."
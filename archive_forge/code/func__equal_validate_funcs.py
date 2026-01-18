from __future__ import annotations
import math
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, TypeVar, cast, overload
from cachetools import TTLCache
from typing_extensions import TypeAlias
import streamlit as st
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
def _equal_validate_funcs(a: ValidateFunc | None, b: ValidateFunc | None) -> bool:
    """True if the two validate functions are equal for the purposes of
    determining whether a given function cache needs to be recreated.
    """
    return a is None and b is None or (a is not None and b is not None)
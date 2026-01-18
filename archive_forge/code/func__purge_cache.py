from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def _purge_cache() -> None:
    """Purge the cache."""
    _cached_css_compile.cache_clear()
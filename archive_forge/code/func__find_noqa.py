from __future__ import annotations
import functools
import linecache
import logging
from typing import Match
from typing import NamedTuple
from flake8 import defaults
from flake8 import utils
@functools.lru_cache(maxsize=512)
def _find_noqa(physical_line: str) -> Match[str] | None:
    return defaults.NOQA_INLINE_REGEXP.search(physical_line)
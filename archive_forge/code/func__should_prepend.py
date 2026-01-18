from __future__ import annotations
from functools import lru_cache
import collections
import enum
import os
import re
import typing as T
@classmethod
@lru_cache(maxsize=None)
def _should_prepend(cls, arg: str) -> bool:
    return arg.startswith(cls.prepend_prefixes)
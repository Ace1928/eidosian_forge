from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
def _strip_inline(self):
    """
        Search for the earliest prefix at the beginning of the line or following a space.
        """
    matcher = re.compile('|'.join((f'(^|\\s)({re.escape(prefix)})' for prefix in self.prefixes.inline)) or '(?!)')
    match = matcher.search(self)
    return self[:match.start() if match else None].strip()
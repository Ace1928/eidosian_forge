from __future__ import annotations
import os
from itertools import chain
from .connection import Resource
from .messaging import Producer
from .utils.collections import EqualityDict
from .utils.compat import register_after_fork
from .utils.functional import lazy
def _all_pools():
    return chain(*(g.values() if g else iter([]) for g in _groups))
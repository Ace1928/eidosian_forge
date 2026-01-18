from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
class MemorySplitProtocol(Protocol):
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]

    def _allocate(self, block: Allocation, is_last: bool) -> bool:
        ...
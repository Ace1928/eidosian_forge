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
class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""
    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value):
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value
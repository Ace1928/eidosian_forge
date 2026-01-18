from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def _is_singleton(s):
    if not isinstance(s, torch.SymInt):
        return False
    if s.node.singleton_int() is not None:
        return True
    return s.node.is_symbolic() and s.node.hint is not None and isinstance(s.node.hint, torch.SymInt) and (s.node.hint.node.singleton_int() is not None)
import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def _to_symtype(t):
    if t is bool:
        return SymBool
    if t is int:
        return SymInt
    if t is float:
        return SymFloat
    return t
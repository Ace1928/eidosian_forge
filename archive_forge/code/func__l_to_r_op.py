from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
def _l_to_r_op(op: F) -> F:
    """Swap the argument order to turn an l-op into an r-op."""

    def r_op(obj: t.Any, other: t.Any) -> t.Any:
        return op(other, obj)
    return t.cast(F, r_op)
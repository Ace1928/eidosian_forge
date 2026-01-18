import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _determine_attrs_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)
    if cmp is not None:
        return (cmp, cmp)
    if eq is None:
        eq = default_eq
    if order is None:
        order = eq
    if eq is False and order is True:
        msg = '`order` can only be True if `eq` is True too.'
        raise ValueError(msg)
    return (eq, order)
from __future__ import annotations
import typing
import numpy as np
from mizani.bounds import expand_range_distinct
from .._utils import ignore_warnings
from ..iapi import range_view
def _expand_range_distinct(x: TupleFloat2, expand: TupleFloat2 | TupleFloat4) -> TupleFloat2:
    a, b = x
    if a > b:
        b, a = expand_range_distinct((b, a), expand)
    else:
        a, b = expand_range_distinct((a, b), expand)
    return (a, b)
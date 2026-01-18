from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def _datetrunc_range(date: datetime.date, unit: str, dialect: Dialect) -> t.Optional[DateRange]:
    """
    Get the date range for a DATE_TRUNC equality comparison:

    Example:
        _datetrunc_range(date(2021-01-01), 'year') == (date(2021-01-01), date(2022-01-01))
    Returns:
        tuple of [min, max) or None if a value can never be equal to `date` for `unit`
    """
    floor = date_floor(date, unit, dialect)
    if date != floor:
        return None
    return (floor, floor + interval(unit))
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
def _simplify_binary(expression, a, b):
    if isinstance(expression, COMPARISONS):
        a = _simplify_integer_cast(a)
        b = _simplify_integer_cast(b)
    if isinstance(expression, exp.Is):
        if isinstance(b, exp.Not):
            c = b.this
            not_ = True
        else:
            c = b
            not_ = False
        if is_null(c):
            if isinstance(a, exp.Literal):
                return exp.true() if not_ else exp.false()
            if is_null(a):
                return exp.false() if not_ else exp.true()
    elif isinstance(expression, NULL_OK):
        return None
    elif is_null(a) or is_null(b):
        return exp.null()
    if a.is_number and b.is_number:
        num_a = int(a.name) if a.is_int else Decimal(a.name)
        num_b = int(b.name) if b.is_int else Decimal(b.name)
        if isinstance(expression, exp.Add):
            return exp.Literal.number(num_a + num_b)
        if isinstance(expression, exp.Mul):
            return exp.Literal.number(num_a * num_b)
        if isinstance(expression, exp.Sub):
            return exp.Literal.number(num_a - num_b) if a.parent is b.parent else None
        if isinstance(expression, exp.Div):
            if isinstance(num_a, int) and isinstance(num_b, int) or a.parent is not b.parent:
                return None
            return exp.Literal.number(num_a / num_b)
        boolean = eval_boolean(expression, num_a, num_b)
        if boolean:
            return boolean
    elif a.is_string and b.is_string:
        boolean = eval_boolean(expression, a.this, b.this)
        if boolean:
            return boolean
    elif _is_date_literal(a) and isinstance(b, exp.Interval):
        date, b = (extract_date(a), extract_interval(b))
        if date and b:
            if isinstance(expression, (exp.Add, exp.DateAdd, exp.DatetimeAdd)):
                return date_literal(date + b, extract_type(a))
            if isinstance(expression, (exp.Sub, exp.DateSub, exp.DatetimeSub)):
                return date_literal(date - b, extract_type(a))
    elif isinstance(a, exp.Interval) and _is_date_literal(b):
        a, date = (extract_interval(a), extract_date(b))
        if a and b and isinstance(expression, exp.Add):
            return date_literal(a + date, extract_type(b))
    elif _is_date_literal(a) and _is_date_literal(b):
        if isinstance(expression, exp.Predicate):
            a, b = (extract_date(a), extract_date(b))
            boolean = eval_boolean(expression, a, b)
            if boolean:
                return boolean
    return None
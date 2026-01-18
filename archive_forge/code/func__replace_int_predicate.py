from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def _replace_int_predicate(expression: exp.Expression) -> None:
    if isinstance(expression, exp.Coalesce):
        for child in expression.iter_expressions():
            _replace_int_predicate(child)
    elif expression.type and expression.type.this in exp.DataType.INTEGER_TYPES:
        expression.replace(expression.neq(0))
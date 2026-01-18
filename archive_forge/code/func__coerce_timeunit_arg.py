from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def _coerce_timeunit_arg(arg: exp.Expression, unit: t.Optional[exp.Expression]) -> exp.Expression:
    if not arg.type:
        return arg
    if arg.type.this in exp.DataType.TEXT_TYPES:
        date_text = arg.name
        is_iso_date_ = is_iso_date(date_text)
        if is_iso_date_ and is_date_unit(unit):
            return arg.replace(exp.cast(arg.copy(), to=exp.DataType.Type.DATE))
        if is_iso_date_ or is_iso_datetime(date_text):
            return arg.replace(exp.cast(arg.copy(), to=exp.DataType.Type.DATETIME))
    elif arg.type.this == exp.DataType.Type.DATE and (not is_date_unit(unit)):
        return arg.replace(exp.cast(arg.copy(), to=exp.DataType.Type.DATETIME))
    return arg
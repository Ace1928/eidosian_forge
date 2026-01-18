from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_eomonth(args: t.List) -> exp.LastDay:
    date = exp.TsOrDsToDate(this=seq_get(args, 0))
    month_lag = seq_get(args, 1)
    if month_lag is None:
        this: exp.Expression = date
    else:
        unit = DATE_DELTA_INTERVAL.get('month')
        this = exp.DateAdd(this=date, expression=month_lag, unit=unit and exp.var(unit))
    return exp.LastDay(this=this)
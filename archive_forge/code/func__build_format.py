from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_format(args: t.List) -> exp.NumberToStr | exp.TimeToStr:
    this = seq_get(args, 0)
    fmt = seq_get(args, 1)
    culture = seq_get(args, 2)
    number_fmt = fmt and (fmt.name in TRANSPILE_SAFE_NUMBER_FMT or not DATE_FMT_RE.search(fmt.name))
    if number_fmt:
        return exp.NumberToStr(this=this, format=fmt, culture=culture)
    if fmt:
        fmt = exp.Literal.string(format_time(fmt.name, TSQL.FORMAT_TIME_MAPPING) if len(fmt.name) == 1 else format_time(fmt.name, TSQL.TIME_MAPPING))
    return exp.TimeToStr(this=this, format=fmt, culture=culture)
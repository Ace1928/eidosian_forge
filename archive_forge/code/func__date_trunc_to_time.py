from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _date_trunc_to_time(args: t.List) -> exp.DateTrunc | exp.TimestampTrunc:
    trunc = date_trunc_to_time(args)
    trunc.set('unit', _map_date_part(trunc.args['unit']))
    return trunc
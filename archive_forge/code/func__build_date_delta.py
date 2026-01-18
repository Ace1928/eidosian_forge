from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.postgres import Postgres
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _build_date_delta(expr_type: t.Type[E]) -> t.Callable[[t.List], E]:

    def _builder(args: t.List) -> E:
        expr = expr_type(this=seq_get(args, 2), expression=seq_get(args, 1), unit=seq_get(args, 0))
        if expr_type is exp.TsOrDsAdd:
            expr.set('return_type', exp.DataType.build('TIMESTAMP'))
        return expr
    return _builder
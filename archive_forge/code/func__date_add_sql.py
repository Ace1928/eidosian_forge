from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _date_add_sql(kind: str) -> t.Callable[[Postgres.Generator, DATE_ADD_OR_SUB], str]:

    def func(self: Postgres.Generator, expression: DATE_ADD_OR_SUB) -> str:
        if isinstance(expression, exp.TsOrDsAdd):
            expression = ts_or_ds_add_cast(expression)
        this = self.sql(expression, 'this')
        unit = expression.args.get('unit')
        expression = self._simplify_unless_literal(expression.expression)
        if not isinstance(expression, exp.Literal):
            self.unsupported('Cannot add non literal')
        expression.args['is_string'] = True
        return f'{this} {kind} {self.sql(exp.Interval(this=expression, unit=unit))}'
    return func
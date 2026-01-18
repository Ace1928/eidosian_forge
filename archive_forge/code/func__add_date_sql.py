from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _add_date_sql(self: Hive.Generator, expression: DATE_ADD_OR_SUB) -> str:
    if isinstance(expression, exp.TsOrDsAdd) and (not expression.unit):
        return self.func('DATE_ADD', expression.this, expression.expression)
    unit = expression.text('unit').upper()
    func, multiplier = DATE_DELTA_INTERVAL.get(unit, ('DATE_ADD', 1))
    if isinstance(expression, exp.DateSub):
        multiplier *= -1
    if expression.expression.is_number:
        modified_increment = exp.Literal.number(int(expression.text('expression')) * multiplier)
    else:
        modified_increment = expression.expression
        if multiplier != 1:
            modified_increment = exp.Mul(this=modified_increment, expression=exp.Literal.number(multiplier))
    return self.func(func, expression.this, modified_increment)
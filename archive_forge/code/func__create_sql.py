from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _create_sql(self: BigQuery.Generator, expression: exp.Create) -> str:
    returns = expression.find(exp.ReturnsProperty)
    if expression.kind == 'FUNCTION' and returns and returns.args.get('is_table'):
        expression.set('kind', 'TABLE FUNCTION')
        if isinstance(expression.expression, (exp.Subquery, exp.Literal)):
            expression.set('expression', expression.expression.this)
    return self.create_sql(expression)
from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _explode_to_unnest_sql(self: Presto.Generator, expression: exp.Lateral) -> str:
    if isinstance(expression.this, exp.Explode):
        return self.sql(exp.Join(this=exp.Unnest(expressions=[expression.this.this], alias=expression.args.get('alias'), offset=isinstance(expression.this, exp.Posexplode)), kind='cross'))
    return self.lateral_sql(expression)
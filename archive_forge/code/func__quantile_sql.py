from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _quantile_sql(self: Presto.Generator, expression: exp.Quantile) -> str:
    self.unsupported('Presto does not support exact quantiles')
    return self.func('APPROX_PERCENTILE', expression.this, expression.args.get('quantile'))
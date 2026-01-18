from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _schema_sql(self: Presto.Generator, expression: exp.Schema) -> str:
    if isinstance(expression.parent, exp.Property):
        columns = ', '.join((f"'{c.name}'" for c in expression.expressions))
        return f'ARRAY[{columns}]'
    if expression.parent:
        for schema in expression.parent.find_all(exp.Schema):
            column_defs = schema.find_all(exp.ColumnDef)
            if column_defs and isinstance(schema.parent, exp.Property):
                expression.expressions.extend(column_defs)
    return self.schema_sql(expression)
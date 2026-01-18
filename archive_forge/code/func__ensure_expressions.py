from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.helper import ensure_list
def _ensure_expressions(values: t.List[NORMALIZE_INPUT]) -> t.List[exp.Expression]:
    results = []
    for value in values:
        if isinstance(value, str):
            results.append(Column.ensure_col(value).expression)
        elif isinstance(value, Column):
            results.append(value.expression)
        elif isinstance(value, exp.Expression):
            results.append(value)
        else:
            raise ValueError(f'Got an invalid type to normalize: {type(value)}')
    return results
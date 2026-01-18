from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def _add_replace_columns(expression: exp.Expression, tables, replace_columns: t.Dict[int, t.Dict[str, str]]) -> None:
    replace = expression.args.get('replace')
    if not replace:
        return
    columns = {e.this.name: e.alias for e in replace}
    for table in tables:
        replace_columns[id(table)] = columns
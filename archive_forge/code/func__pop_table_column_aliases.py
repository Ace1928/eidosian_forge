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
def _pop_table_column_aliases(derived_tables: t.List[exp.CTE | exp.Subquery]) -> None:
    """
    Remove table column aliases.

    For example, `col1` and `col2` will be dropped in SELECT ... FROM (SELECT ...) AS foo(col1, col2)
    """
    for derived_table in derived_tables:
        if isinstance(derived_table.parent, exp.With) and derived_table.parent.recursive:
            continue
        table_alias = derived_table.args.get('alias')
        if table_alias:
            table_alias.args.pop('columns', None)
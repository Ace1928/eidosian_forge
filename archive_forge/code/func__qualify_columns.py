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
def _qualify_columns(scope: Scope, resolver: Resolver) -> None:
    """Disambiguate columns, ensuring each column specifies a source"""
    for column in scope.columns:
        column_table = column.table
        column_name = column.name
        if column_table and column_table in scope.sources:
            source_columns = resolver.get_source_columns(column_table)
            if source_columns and column_name not in source_columns and ('*' not in source_columns):
                raise OptimizeError(f'Unknown column: {column_name}')
        if not column_table:
            if scope.pivots and (not column.find_ancestor(exp.Pivot)):
                column.set('table', exp.to_identifier(scope.pivots[0].alias))
                continue
            column_table = resolver.get_table(column_name)
            if column_table:
                column.set('table', column_table)
        elif column_table not in scope.sources and (not scope.parent or column_table not in scope.parent.sources or (not scope.is_correlated_subquery)):
            root, *parts = column.parts
            if root.name in scope.sources:
                column_table = root
                root, *parts = parts
            else:
                column_table = resolver.get_table(root.name)
            if column_table:
                column.replace(exp.Dot.build([exp.column(root, table=column_table), *parts]))
    for pivot in scope.pivots:
        for column in pivot.find_all(exp.Column):
            if not column.table and column.name in resolver.all_columns:
                column_table = resolver.get_table(column.name)
                if column_table:
                    column.set('table', column_table)
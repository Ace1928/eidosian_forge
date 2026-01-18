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
def _expand_using(scope: Scope, resolver: Resolver) -> t.Dict[str, t.Any]:
    joins = list(scope.find_all(exp.Join))
    names = {join.alias_or_name for join in joins}
    ordered = [key for key in scope.selected_sources if key not in names]
    column_tables: t.Dict[str, t.Dict[str, t.Any]] = {}
    for join in joins:
        using = join.args.get('using')
        if not using:
            continue
        join_table = join.alias_or_name
        columns = {}
        for source_name in scope.selected_sources:
            if source_name in ordered:
                for column_name in resolver.get_source_columns(source_name):
                    if column_name not in columns:
                        columns[column_name] = source_name
        source_table = ordered[-1]
        ordered.append(join_table)
        join_columns = resolver.get_source_columns(join_table)
        conditions = []
        for identifier in using:
            identifier = identifier.name
            table = columns.get(identifier)
            if not table or identifier not in join_columns:
                if (columns and '*' not in columns) and join_columns:
                    raise OptimizeError(f'Cannot automatically join: {identifier}')
            table = table or source_table
            conditions.append(exp.column(identifier, table=table).eq(exp.column(identifier, table=join_table)))
            tables = column_tables.setdefault(identifier, {})
            if table not in tables:
                tables[table] = None
            if join_table not in tables:
                tables[join_table] = None
        join.args.pop('using')
        join.set('on', exp.and_(*conditions, copy=False))
    if column_tables:
        for column in scope.columns:
            if not column.table and column.name in column_tables:
                tables = column_tables[column.name]
                coalesce = [exp.column(column.name, table=table) for table in tables]
                replacement = exp.Coalesce(this=coalesce[0], expressions=coalesce[1:])
                if isinstance(column.parent, exp.Select):
                    replacement = alias(replacement, alias=column.name, copy=False)
                scope.replace(column, replacement)
    return column_tables
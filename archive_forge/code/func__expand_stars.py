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
def _expand_stars(scope: Scope, resolver: Resolver, using_column_tables: t.Dict[str, t.Any], pseudocolumns: t.Set[str]) -> None:
    """Expand stars to lists of column selections"""
    new_selections = []
    except_columns: t.Dict[int, t.Set[str]] = {}
    replace_columns: t.Dict[int, t.Dict[str, str]] = {}
    coalesced_columns = set()
    pivot_output_columns = None
    pivot_exclude_columns = None
    pivot = t.cast(t.Optional[exp.Pivot], seq_get(scope.pivots, 0))
    if isinstance(pivot, exp.Pivot) and (not pivot.alias_column_names):
        if pivot.unpivot:
            pivot_output_columns = [c.output_name for c in _unpivot_columns(pivot)]
            field = pivot.args.get('field')
            if isinstance(field, exp.In):
                pivot_exclude_columns = {c.output_name for e in field.expressions for c in e.find_all(exp.Column)}
        else:
            pivot_exclude_columns = set((c.output_name for c in pivot.find_all(exp.Column)))
            pivot_output_columns = [c.output_name for c in pivot.args.get('columns', [])]
            if not pivot_output_columns:
                pivot_output_columns = [c.alias_or_name for c in pivot.expressions]
    for expression in scope.expression.selects:
        if isinstance(expression, exp.Star):
            tables = list(scope.selected_sources)
            _add_except_columns(expression, tables, except_columns)
            _add_replace_columns(expression, tables, replace_columns)
        elif expression.is_star and (not isinstance(expression, exp.Dot)):
            tables = [expression.table]
            _add_except_columns(expression.this, tables, except_columns)
            _add_replace_columns(expression.this, tables, replace_columns)
        else:
            new_selections.append(expression)
            continue
        for table in tables:
            if table not in scope.sources:
                raise OptimizeError(f'Unknown table: {table}')
            columns = resolver.get_source_columns(table, only_visible=True)
            columns = columns or scope.outer_columns
            if pseudocolumns:
                columns = [name for name in columns if name.upper() not in pseudocolumns]
            if not columns or '*' in columns:
                return
            table_id = id(table)
            columns_to_exclude = except_columns.get(table_id) or set()
            if pivot:
                if pivot_output_columns and pivot_exclude_columns:
                    pivot_columns = [c for c in columns if c not in pivot_exclude_columns]
                    pivot_columns.extend(pivot_output_columns)
                else:
                    pivot_columns = pivot.alias_column_names
                if pivot_columns:
                    new_selections.extend((alias(exp.column(name, table=pivot.alias), name, copy=False) for name in pivot_columns if name not in columns_to_exclude))
                    continue
            for name in columns:
                if name in columns_to_exclude or name in coalesced_columns:
                    continue
                if name in using_column_tables and table in using_column_tables[name]:
                    coalesced_columns.add(name)
                    tables = using_column_tables[name]
                    coalesce = [exp.column(name, table=table) for table in tables]
                    new_selections.append(alias(exp.Coalesce(this=coalesce[0], expressions=coalesce[1:]), alias=name, copy=False))
                else:
                    alias_ = replace_columns.get(table_id, {}).get(name, name)
                    column = exp.column(name, table=table)
                    new_selections.append(alias(column, alias_, copy=False) if alias_ != name else column)
    if new_selections and isinstance(scope.expression, exp.Select):
        scope.expression.set('expressions', new_selections)
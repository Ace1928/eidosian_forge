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
def _get_unambiguous_columns(self, source_columns: t.Dict[str, t.Sequence[str]]) -> t.Mapping[str, str]:
    """
        Find all the unambiguous columns in sources.

        Args:
            source_columns: Mapping of names to source columns.

        Returns:
            Mapping of column name to source name.
        """
    if not source_columns:
        return {}
    source_columns_pairs = list(source_columns.items())
    first_table, first_columns = source_columns_pairs[0]
    if len(source_columns_pairs) == 1:
        return SingleValuedMapping(first_columns, first_table)
    unambiguous_columns = {col: first_table for col in first_columns}
    all_columns = set(unambiguous_columns)
    for table, columns in source_columns_pairs[1:]:
        unique = set(columns)
        ambiguous = all_columns.intersection(unique)
        all_columns.update(columns)
        for column in ambiguous:
            unambiguous_columns.pop(column, None)
        for column in unique.difference(ambiguous):
            unambiguous_columns[column] = table
    return unambiguous_columns
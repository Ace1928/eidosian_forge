from __future__ import annotations
import functools
import logging
import typing as t
import zlib
from copy import copy
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.group import GroupedData
from sqlglot.dataframe.sql.normalize import normalize
from sqlglot.dataframe.sql.operations import Operation, operation
from sqlglot.dataframe.sql.readwriter import DataFrameWriter
from sqlglot.dataframe.sql.transforms import replace_id_value
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.dataframe.sql.window import Window
from sqlglot.helper import ensure_list, object_to_dict, seq_get
def _resolve_pending_hints(self) -> DataFrame:
    df = self.copy()
    if not self.pending_hints:
        return df
    expression = df.expression
    hint_expression = expression.args.get('hint') or exp.Hint(expressions=[])
    for hint in df.pending_partition_hints:
        hint_expression.append('expressions', hint)
        df.pending_hints.remove(hint)
    join_aliases = {join_table.alias_or_name for join_table in get_tables_from_expression_with_join(expression)}
    if join_aliases:
        for hint in df.pending_join_hints:
            for sequence_id_expression in hint.expressions:
                sequence_id_or_name = sequence_id_expression.alias_or_name
                sequence_ids_to_match = [sequence_id_or_name]
                if sequence_id_or_name in df.spark.name_to_sequence_id_mapping:
                    sequence_ids_to_match = df.spark.name_to_sequence_id_mapping[sequence_id_or_name]
                matching_ctes = [cte for cte in reversed(expression.ctes) if cte.args['sequence_id'] in sequence_ids_to_match]
                for matching_cte in matching_ctes:
                    if matching_cte.alias_or_name in join_aliases:
                        sequence_id_expression.set('this', matching_cte.args['alias'].this)
                        df.pending_hints.remove(hint)
                        break
            hint_expression.append('expressions', hint)
    if hint_expression.expressions:
        expression.set('hint', hint_expression)
    return df
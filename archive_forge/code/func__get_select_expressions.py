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
def _get_select_expressions(self) -> t.List[t.Tuple[t.Union[t.Type[exp.Cache], OutputExpressionContainer], exp.Select]]:
    select_expressions: t.List[t.Tuple[t.Union[t.Type[exp.Cache], OutputExpressionContainer], exp.Select]] = []
    main_select_ctes: t.List[exp.CTE] = []
    for cte in self.expression.ctes:
        cache_storage_level = cte.args.get('cache_storage_level')
        if cache_storage_level:
            select_expression = cte.this.copy()
            select_expression.set('with', exp.With(expressions=copy(main_select_ctes)))
            select_expression.set('cte_alias_name', cte.alias_or_name)
            select_expression.set('cache_storage_level', cache_storage_level)
            select_expressions.append((exp.Cache, select_expression))
        else:
            main_select_ctes.append(cte)
    main_select = self.expression.copy()
    if main_select_ctes:
        main_select.set('with', exp.With(expressions=main_select_ctes))
    expression_select_pair = (type(self.output_expression_container), main_select)
    select_expressions.append(expression_select_pair)
    return select_expressions
from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.helper import ensure_list
def _set_alias_name(id: exp.Identifier, name: str):
    id.set('this', name)
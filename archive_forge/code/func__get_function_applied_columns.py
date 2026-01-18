from __future__ import annotations
import typing as t
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.operations import Operation, operation
def _get_function_applied_columns(self, func_name: str, cols: t.Tuple[str, ...]) -> t.List[Column]:
    func_name = func_name.lower()
    return [getattr(F, func_name)(name).alias(f'{func_name}({name})') for name in cols]
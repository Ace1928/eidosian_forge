from typing import Any, Optional
import pyarrow as pa
from fugue.column.expressions import (
from triad import Schema
class _UnaryAggFuncExpr(_FuncExpr):

    def __init__(self, func: str, col: ColumnExpr, arg_distinct: bool=False):
        super().__init__(func, col, arg_distinct=arg_distinct)

    def infer_alias(self) -> ColumnExpr:
        return self if self.output_name != '' else self.alias(self.args[0].infer_alias().output_name)

    def _copy(self) -> _FuncExpr:
        return _UnaryAggFuncExpr(self.func, *self.args, **self.kwargs)
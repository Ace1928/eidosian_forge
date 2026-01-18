from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
def _get_column_mentions(column: ColumnExpr) -> Iterable[str]:
    if isinstance(column, _NamedColumnExpr):
        yield column.name
    elif isinstance(column, _FuncExpr):
        for a in column.args:
            yield from _get_column_mentions(a)
        for a in column.kwargs.values():
            yield from _get_column_mentions(a)
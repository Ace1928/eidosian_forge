from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def _on_common_unary(self, expr: _UnaryOpExpr) -> Iterable[str]:
    if expr.op == '-':
        yield expr.op
        yield from self._generate(expr.col, bracket=True)
    elif expr.op == '~':
        yield 'NOT '
        yield from self._generate(expr.col, bracket=True)
    elif expr.op == 'IS_NULL':
        yield from self._generate(expr.col, bracket=True)
        yield ' IS NULL'
    elif expr.op == 'NOT_NULL':
        yield from self._generate(expr.col, bracket=True)
        yield ' IS NOT NULL'
    else:
        raise NotImplementedError(expr)
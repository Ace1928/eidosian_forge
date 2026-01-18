from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def _get_cols() -> Iterable[ColumnExpr]:
    for c in self.all_cols:
        if isinstance(c, _WildcardExpr):
            yield from [col(n) for n in schema.names]
        else:
            yield c
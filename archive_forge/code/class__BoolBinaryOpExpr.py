from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _BoolBinaryOpExpr(_BinaryOpExpr):

    def _copy(self) -> _FuncExpr:
        return _BoolBinaryOpExpr(self.op, self.left, self.right)

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        return self.as_type or pa.bool_()
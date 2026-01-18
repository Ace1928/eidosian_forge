from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _WildcardExpr(ColumnExpr):

    @property
    def body_str(self) -> str:
        return '*'

    @property
    def name(self) -> str:
        raise NotImplementedError("wildcard column doesn't have a name")

    @property
    def output_name(self) -> str:
        raise NotImplementedError("wildcard column doesn't have an output_name")

    def __uuid__(self) -> str:
        """The unique id of this instance

        :return: the unique id
        """
        return to_uuid(str(type(self)))
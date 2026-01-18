from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
class aggregate_strings(GenericFunction[str]):
    """Implement a generic string aggregation function.

    This function will concatenate non-null values into a string and
    separate the values by a delimiter.

    This function is compiled on a per-backend basis, into functions
    such as ``group_concat()``, ``string_agg()``, or ``LISTAGG()``.

    e.g. Example usage with delimiter '.'::

        stmt = select(func.aggregate_strings(table.c.str_col, "."))

    The return type of this function is :class:`.String`.

    .. versionadded: 2.0.21

    """
    type = sqltypes.String()
    _has_args = True
    inherit_cache = True

    def __init__(self, clause: _ColumnExpressionArgument[Any], separator: str):
        super().__init__(clause, separator)
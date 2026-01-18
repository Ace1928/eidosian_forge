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
class grouping_sets(GenericFunction[_T]):
    """Implement the ``GROUPING SETS`` grouping operation.

    This function is used as part of the GROUP BY of a statement,
    e.g. :meth:`_expression.Select.group_by`::

        stmt = select(
            func.sum(table.c.value), table.c.col_1, table.c.col_2
        ).group_by(func.grouping_sets(table.c.col_1, table.c.col_2))

    In order to group by multiple sets, use the :func:`.tuple_` construct::

        from sqlalchemy import tuple_

        stmt = select(
            func.sum(table.c.value),
            table.c.col_1, table.c.col_2,
            table.c.col_3
        ).group_by(
            func.grouping_sets(
                tuple_(table.c.col_1, table.c.col_2),
                tuple_(table.c.value, table.c.col_3),
            )
        )


    .. versionadded:: 1.2

    """
    _has_args = True
    inherit_cache = True
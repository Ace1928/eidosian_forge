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
class array_agg(GenericFunction[_T]):
    """Support for the ARRAY_AGG function.

    The ``func.array_agg(expr)`` construct returns an expression of
    type :class:`_types.ARRAY`.

    e.g.::

        stmt = select(func.array_agg(table.c.values)[2:5])

    .. seealso::

        :func:`_postgresql.array_agg` - PostgreSQL-specific version that
        returns :class:`_postgresql.ARRAY`, which has PG-specific operators
        added.

    """
    inherit_cache = True

    def __init__(self, *args: _ColumnExpressionArgument[Any], **kwargs: Any):
        fn_args: Sequence[ColumnElement[Any]] = [coercions.expect(roles.ExpressionElementRole, c, apply_propagate_attrs=self) for c in args]
        default_array_type = kwargs.pop('_default_array_type', sqltypes.ARRAY)
        if 'type_' not in kwargs:
            type_from_args = _type_from_args(fn_args)
            if isinstance(type_from_args, sqltypes.ARRAY):
                kwargs['type_'] = type_from_args
            else:
                kwargs['type_'] = default_array_type(type_from_args, dimensions=1)
        kwargs['_parsed_args'] = fn_args
        super().__init__(*fn_args, **kwargs)
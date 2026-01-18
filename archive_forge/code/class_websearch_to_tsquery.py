from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal
class websearch_to_tsquery(_regconfig_fn):
    """The PostgreSQL ``websearch_to_tsquery`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_postgresql.TSQUERY`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.websearch_to_tsquery` will be used automatically when
    invoking ``sqlalchemy.func.websearch_to_tsquery()``, ensuring the correct
    argument and return type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """
    inherit_cache = True
    type = types.TSQUERY
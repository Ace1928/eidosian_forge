from __future__ import annotations
from typing import Any
from .._typing import _OnConflictIndexElementsT
from .._typing import _OnConflictIndexWhereT
from .._typing import _OnConflictSetT
from .._typing import _OnConflictWhereT
from ... import util
from ...sql import coercions
from ...sql import roles
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...util.typing import Self

        Specifies a DO NOTHING action for ON CONFLICT clause.

        :param index_elements:
         A sequence consisting of string column names, :class:`_schema.Column`
         objects, or other column expression objects that will be used
         to infer a target index or unique constraint.

        :param index_where:
         Additional WHERE criterion that can be used to infer a
         conditional target index.

        
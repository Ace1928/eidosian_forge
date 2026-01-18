from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class Computed(FetchedValue, SchemaItem):
    """Defines a generated column, i.e. "GENERATED ALWAYS AS" syntax.

    The :class:`.Computed` construct is an inline construct added to the
    argument list of a :class:`_schema.Column` object::

        from sqlalchemy import Computed

        Table('square', metadata_obj,
            Column('side', Float, nullable=False),
            Column('area', Float, Computed('side * side'))
        )

    See the linked documentation below for complete details.

    .. versionadded:: 1.3.11

    .. seealso::

        :ref:`computed_ddl`

    """
    __visit_name__ = 'computed_column'
    column: Optional[Column[Any]]

    @_document_text_coercion('sqltext', ':class:`.Computed`', ':paramref:`.Computed.sqltext`')
    def __init__(self, sqltext: _DDLColumnArgument, persisted: Optional[bool]=None) -> None:
        """Construct a GENERATED ALWAYS AS DDL construct to accompany a
        :class:`_schema.Column`.

        :param sqltext:
          A string containing the column generation expression, which will be
          used verbatim, or a SQL expression construct, such as a
          :func:`_expression.text`
          object.   If given as a string, the object is converted to a
          :func:`_expression.text` object.

        :param persisted:
          Optional, controls how this column should be persisted by the
          database.   Possible values are:

          * ``None``, the default, it will use the default persistence
            defined by the database.
          * ``True``, will render ``GENERATED ALWAYS AS ... STORED``, or the
            equivalent for the target database if supported.
          * ``False``, will render ``GENERATED ALWAYS AS ... VIRTUAL``, or
            the equivalent for the target database if supported.

          Specifying ``True`` or ``False`` may raise an error when the DDL
          is emitted to the target database if the database does not support
          that persistence option.   Leaving this parameter at its default
          of ``None`` is guaranteed to succeed for all databases that support
          ``GENERATED ALWAYS AS``.

        """
        self.sqltext = coercions.expect(roles.DDLExpressionRole, sqltext)
        self.persisted = persisted
        self.column = None

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, Column)
        if not isinstance(parent.server_default, (type(None), Computed)) or not isinstance(parent.server_onupdate, (type(None), Computed)):
            raise exc.ArgumentError('A generated column cannot specify a server_default or a server_onupdate argument')
        self.column = parent
        parent.computed = self
        self.column.server_onupdate = self
        self.column.server_default = self

    def _as_for_update(self, for_update: bool) -> FetchedValue:
        return self

    @util.deprecated('1.4', 'The :meth:`_schema.Computed.copy` method is deprecated and will be removed in a future release.')
    def copy(self, *, target_table: Optional[Table]=None, **kw: Any) -> Computed:
        return self._copy(target_table=target_table, **kw)

    def _copy(self, *, target_table: Optional[Table]=None, **kw: Any) -> Computed:
        sqltext = _copy_expression(self.sqltext, self.column.table if self.column is not None else None, target_table)
        g = Computed(sqltext, persisted=self.persisted)
        return self._schema_item_copy(g)
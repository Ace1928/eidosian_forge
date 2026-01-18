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
@inspection._self_inspects
class FetchedValue(SchemaEventTarget):
    """A marker for a transparent database-side default.

    Use :class:`.FetchedValue` when the database is configured
    to provide some automatic default for a column.

    E.g.::

        Column('foo', Integer, FetchedValue())

    Would indicate that some trigger or default generator
    will create a new value for the ``foo`` column during an
    INSERT.

    .. seealso::

        :ref:`triggered_columns`

    """
    is_server_default = True
    reflected = False
    has_argument = False
    is_clause_element = False
    is_identity = False
    column: Optional[Column[Any]]

    def __init__(self, for_update: bool=False) -> None:
        self.for_update = for_update

    def _as_for_update(self, for_update: bool) -> FetchedValue:
        if for_update == self.for_update:
            return self
        else:
            return self._clone(for_update)

    def _copy(self) -> FetchedValue:
        return FetchedValue(self.for_update)

    def _clone(self, for_update: bool) -> Self:
        n = self.__class__.__new__(self.__class__)
        n.__dict__.update(self.__dict__)
        n.__dict__.pop('column', None)
        n.for_update = for_update
        return n

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        column = parent
        assert isinstance(column, Column)
        self.column = column
        if self.for_update:
            self.column.server_onupdate = self
        else:
            self.column.server_default = self

    def __repr__(self) -> str:
        return util.generic_repr(self)
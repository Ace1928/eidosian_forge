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
class ColumnCollectionConstraint(ColumnCollectionMixin, Constraint):
    """A constraint that proxies a ColumnCollection."""

    def __init__(self, *columns: _DDLColumnArgument, name: _ConstraintNameArgument=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, info: Optional[_InfoType]=None, _autoattach: bool=True, _column_flag: bool=False, _gather_expressions: Optional[List[_DDLColumnArgument]]=None, **dialect_kw: Any) -> None:
        """
        :param \\*columns:
          A sequence of column names or Column objects.

        :param name:
          Optional, the in-database name of this constraint.

        :param deferrable:
          Optional bool.  If set, emit DEFERRABLE or NOT DEFERRABLE when
          issuing DDL for this constraint.

        :param initially:
          Optional string.  If set, emit INITIALLY <value> when issuing DDL
          for this constraint.

        :param \\**dialect_kw: other keyword arguments including
          dialect-specific arguments are propagated to the :class:`.Constraint`
          superclass.

        """
        Constraint.__init__(self, name=name, deferrable=deferrable, initially=initially, info=info, **dialect_kw)
        ColumnCollectionMixin.__init__(self, *columns, _autoattach=_autoattach, _column_flag=_column_flag)
    columns: ReadOnlyColumnCollection[str, Column[Any]]
    'A :class:`_expression.ColumnCollection` representing the set of columns\n    for this constraint.\n\n    '

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, (Column, Table))
        Constraint._set_parent(self, parent)
        ColumnCollectionMixin._set_parent(self, parent)

    def __contains__(self, x: Any) -> bool:
        return x in self._columns

    @util.deprecated('1.4', 'The :meth:`_schema.ColumnCollectionConstraint.copy` method is deprecated and will be removed in a future release.')
    def copy(self, *, target_table: Optional[Table]=None, **kw: Any) -> ColumnCollectionConstraint:
        return self._copy(target_table=target_table, **kw)

    def _copy(self, *, target_table: Optional[Table]=None, **kw: Any) -> ColumnCollectionConstraint:
        constraint_kwargs = {}
        for dialect_name in self.dialect_options:
            dialect_options = self.dialect_options[dialect_name]._non_defaults
            for dialect_option_key, dialect_option_value in dialect_options.items():
                constraint_kwargs[dialect_name + '_' + dialect_option_key] = dialect_option_value
        assert isinstance(self.parent, Table)
        c = self.__class__(*[_copy_expression(expr, self.parent, target_table) for expr in self._columns], name=self.name, deferrable=self.deferrable, initially=self.initially, comment=self.comment, **constraint_kwargs)
        return self._schema_item_copy(c)

    def contains_column(self, col: Column[Any]) -> bool:
        """Return True if this constraint contains the given column.

        Note that this object also contains an attribute ``.columns``
        which is a :class:`_expression.ColumnCollection` of
        :class:`_schema.Column` objects.

        """
        return self._columns.contains_column(col)

    def __iter__(self) -> Iterator[Column[Any]]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)
from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
@Operations.register_operation('create_index')
@BatchOperations.register_operation('create_index', 'batch_create_index')
class CreateIndexOp(MigrateOperation):
    """Represent a create index operation."""

    def __init__(self, index_name: Optional[str], table_name: str, columns: Sequence[Union[str, TextClause, ColumnElement[Any]]], *, schema: Optional[str]=None, unique: bool=False, if_not_exists: Optional[bool]=None, **kw: Any) -> None:
        self.index_name = index_name
        self.table_name = table_name
        self.columns = columns
        self.schema = schema
        self.unique = unique
        self.if_not_exists = if_not_exists
        self.kw = kw

    def reverse(self) -> DropIndexOp:
        return DropIndexOp.from_index(self.to_index())

    def to_diff_tuple(self) -> Tuple[str, Index]:
        return ('add_index', self.to_index())

    @classmethod
    def from_index(cls, index: Index) -> CreateIndexOp:
        assert index.table is not None
        return cls(index.name, index.table.name, index.expressions, schema=index.table.schema, unique=index.unique, **index.kwargs)

    def to_index(self, migration_context: Optional[MigrationContext]=None) -> Index:
        schema_obj = schemaobj.SchemaObjects(migration_context)
        idx = schema_obj.index(self.index_name, self.table_name, self.columns, schema=self.schema, unique=self.unique, **self.kw)
        return idx

    @classmethod
    def create_index(cls, operations: Operations, index_name: Optional[str], table_name: str, columns: Sequence[Union[str, TextClause, Function[Any]]], *, schema: Optional[str]=None, unique: bool=False, if_not_exists: Optional[bool]=None, **kw: Any) -> None:
        """Issue a "create index" instruction using the current
        migration context.

        e.g.::

            from alembic import op

            op.create_index("ik_test", "t1", ["foo", "bar"])

        Functional indexes can be produced by using the
        :func:`sqlalchemy.sql.expression.text` construct::

            from alembic import op
            from sqlalchemy import text

            op.create_index("ik_test", "t1", [text("lower(foo)")])

        :param index_name: name of the index.
        :param table_name: name of the owning table.
        :param columns: a list consisting of string column names and/or
         :func:`~sqlalchemy.sql.expression.text` constructs.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.
        :param unique: If True, create a unique index.

        :param quote: Force quoting of this column's name on or off,
         corresponding to ``True`` or ``False``. When left at its default
         of ``None``, the column identifier will be quoted according to
         whether the name is case sensitive (identifiers with at least one
         upper case character are treated as case sensitive), or if it's a
         reserved word. This flag is only needed to force quoting of a
         reserved word which is not known by the SQLAlchemy dialect.

        :param if_not_exists: If True, adds IF NOT EXISTS operator when
         creating the new index.

         .. versionadded:: 1.12.0

        :param \\**kw: Additional keyword arguments not mentioned above are
         dialect specific, and passed in the form
         ``<dialectname>_<argname>``.
         See the documentation regarding an individual dialect at
         :ref:`dialect_toplevel` for detail on documented arguments.

        """
        op = cls(index_name, table_name, columns, schema=schema, unique=unique, if_not_exists=if_not_exists, **kw)
        return operations.invoke(op)

    @classmethod
    def batch_create_index(cls, operations: BatchOperations, index_name: str, columns: List[str], **kw: Any) -> None:
        """Issue a "create index" instruction using the
        current batch migration context.

        .. seealso::

            :meth:`.Operations.create_index`

        """
        op = cls(index_name, operations.impl.table_name, columns, schema=operations.impl.schema, **kw)
        return operations.invoke(op)
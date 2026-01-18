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
@Operations.register_operation('add_column')
@BatchOperations.register_operation('add_column', 'batch_add_column')
class AddColumnOp(AlterTableOp):
    """Represent an add column operation."""

    def __init__(self, table_name: str, column: Column[Any], *, schema: Optional[str]=None, **kw: Any) -> None:
        super().__init__(table_name, schema=schema)
        self.column = column
        self.kw = kw

    def reverse(self) -> DropColumnOp:
        return DropColumnOp.from_column_and_tablename(self.schema, self.table_name, self.column)

    def to_diff_tuple(self) -> Tuple[str, Optional[str], str, Column[Any]]:
        return ('add_column', self.schema, self.table_name, self.column)

    def to_column(self) -> Column[Any]:
        return self.column

    @classmethod
    def from_column(cls, col: Column[Any]) -> AddColumnOp:
        return cls(col.table.name, col, schema=col.table.schema)

    @classmethod
    def from_column_and_tablename(cls, schema: Optional[str], tname: str, col: Column[Any]) -> AddColumnOp:
        return cls(tname, col, schema=schema)

    @classmethod
    def add_column(cls, operations: Operations, table_name: str, column: Column[Any], *, schema: Optional[str]=None) -> None:
        """Issue an "add column" instruction using the current
        migration context.

        e.g.::

            from alembic import op
            from sqlalchemy import Column, String

            op.add_column("organization", Column("name", String()))

        The :meth:`.Operations.add_column` method typically corresponds
        to the SQL command "ALTER TABLE... ADD COLUMN".    Within the scope
        of this command, the column's name, datatype, nullability,
        and optional server-generated defaults may be indicated.

        .. note::

            With the exception of NOT NULL constraints or single-column FOREIGN
            KEY constraints, other kinds of constraints such as PRIMARY KEY,
            UNIQUE or CHECK constraints **cannot** be generated using this
            method; for these constraints, refer to operations such as
            :meth:`.Operations.create_primary_key` and
            :meth:`.Operations.create_check_constraint`. In particular, the
            following :class:`~sqlalchemy.schema.Column` parameters are
            **ignored**:

            * :paramref:`~sqlalchemy.schema.Column.primary_key` - SQL databases
              typically do not support an ALTER operation that can add
              individual columns one at a time to an existing primary key
              constraint, therefore it's less ambiguous to use the
              :meth:`.Operations.create_primary_key` method, which assumes no
              existing primary key constraint is present.
            * :paramref:`~sqlalchemy.schema.Column.unique` - use the
              :meth:`.Operations.create_unique_constraint` method
            * :paramref:`~sqlalchemy.schema.Column.index` - use the
              :meth:`.Operations.create_index` method


        The provided :class:`~sqlalchemy.schema.Column` object may include a
        :class:`~sqlalchemy.schema.ForeignKey` constraint directive,
        referencing a remote table name. For this specific type of constraint,
        Alembic will automatically emit a second ALTER statement in order to
        add the single-column FOREIGN KEY constraint separately::

            from alembic import op
            from sqlalchemy import Column, INTEGER, ForeignKey

            op.add_column(
                "organization",
                Column("account_id", INTEGER, ForeignKey("accounts.id")),
            )

        The column argument passed to :meth:`.Operations.add_column` is a
        :class:`~sqlalchemy.schema.Column` construct, used in the same way it's
        used in SQLAlchemy. In particular, values or functions to be indicated
        as producing the column's default value on the database side are
        specified using the ``server_default`` parameter, and not ``default``
        which only specifies Python-side defaults::

            from alembic import op
            from sqlalchemy import Column, TIMESTAMP, func

            # specify "DEFAULT NOW" along with the column add
            op.add_column(
                "account",
                Column("timestamp", TIMESTAMP, server_default=func.now()),
            )

        :param table_name: String name of the parent table.
        :param column: a :class:`sqlalchemy.schema.Column` object
         representing the new column.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        """
        op = cls(table_name, column, schema=schema)
        return operations.invoke(op)

    @classmethod
    def batch_add_column(cls, operations: BatchOperations, column: Column[Any], *, insert_before: Optional[str]=None, insert_after: Optional[str]=None) -> None:
        """Issue an "add column" instruction using the current
        batch migration context.

        .. seealso::

            :meth:`.Operations.add_column`

        """
        kw = {}
        if insert_before:
            kw['insert_before'] = insert_before
        if insert_after:
            kw['insert_after'] = insert_after
        op = cls(operations.impl.table_name, column, schema=operations.impl.schema, **kw)
        return operations.invoke(op)
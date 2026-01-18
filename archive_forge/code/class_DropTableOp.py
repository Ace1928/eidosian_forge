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
@Operations.register_operation('drop_table')
class DropTableOp(MigrateOperation):
    """Represent a drop table operation."""

    def __init__(self, table_name: str, *, schema: Optional[str]=None, table_kw: Optional[MutableMapping[Any, Any]]=None, _reverse: Optional[CreateTableOp]=None) -> None:
        self.table_name = table_name
        self.schema = schema
        self.table_kw = table_kw or {}
        self.comment = self.table_kw.pop('comment', None)
        self.info = self.table_kw.pop('info', None)
        self.prefixes = self.table_kw.pop('prefixes', None)
        self._reverse = _reverse

    def to_diff_tuple(self) -> Tuple[str, Table]:
        return ('remove_table', self.to_table())

    def reverse(self) -> CreateTableOp:
        return CreateTableOp.from_table(self.to_table())

    @classmethod
    def from_table(cls, table: Table, *, _namespace_metadata: Optional[MetaData]=None) -> DropTableOp:
        return cls(table.name, schema=table.schema, table_kw={'comment': table.comment, 'info': dict(table.info), 'prefixes': list(table._prefixes), **table.kwargs}, _reverse=CreateTableOp.from_table(table, _namespace_metadata=_namespace_metadata))

    def to_table(self, migration_context: Optional[MigrationContext]=None) -> Table:
        if self._reverse:
            cols_and_constraints = self._reverse.columns
        else:
            cols_and_constraints = []
        schema_obj = schemaobj.SchemaObjects(migration_context)
        t = schema_obj.table(self.table_name, *cols_and_constraints, comment=self.comment, info=self.info.copy() if self.info else {}, prefixes=list(self.prefixes) if self.prefixes else [], schema=self.schema, _constraints_included=self._reverse._constraints_included if self._reverse else False, **self.table_kw)
        return t

    @classmethod
    def drop_table(cls, operations: Operations, table_name: str, *, schema: Optional[str]=None, **kw: Any) -> None:
        """Issue a "drop table" instruction using the current
        migration context.


        e.g.::

            drop_table("accounts")

        :param table_name: Name of the table
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.
        :param \\**kw: Other keyword arguments are passed to the underlying
         :class:`sqlalchemy.schema.Table` object created for the command.

        """
        op = cls(table_name, schema=schema, table_kw=kw)
        operations.invoke(op)
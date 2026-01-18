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
@Operations.register_operation('alter_column')
@BatchOperations.register_operation('alter_column', 'batch_alter_column')
class AlterColumnOp(AlterTableOp):
    """Represent an alter column operation."""

    def __init__(self, table_name: str, column_name: str, *, schema: Optional[str]=None, existing_type: Optional[Any]=None, existing_server_default: Any=False, existing_nullable: Optional[bool]=None, existing_comment: Optional[str]=None, modify_nullable: Optional[bool]=None, modify_comment: Optional[Union[str, Literal[False]]]=False, modify_server_default: Any=False, modify_name: Optional[str]=None, modify_type: Optional[Any]=None, **kw: Any) -> None:
        super().__init__(table_name, schema=schema)
        self.column_name = column_name
        self.existing_type = existing_type
        self.existing_server_default = existing_server_default
        self.existing_nullable = existing_nullable
        self.existing_comment = existing_comment
        self.modify_nullable = modify_nullable
        self.modify_comment = modify_comment
        self.modify_server_default = modify_server_default
        self.modify_name = modify_name
        self.modify_type = modify_type
        self.kw = kw

    def to_diff_tuple(self) -> Any:
        col_diff = []
        schema, tname, cname = (self.schema, self.table_name, self.column_name)
        if self.modify_type is not None:
            col_diff.append(('modify_type', schema, tname, cname, {'existing_nullable': self.existing_nullable, 'existing_server_default': self.existing_server_default, 'existing_comment': self.existing_comment}, self.existing_type, self.modify_type))
        if self.modify_nullable is not None:
            col_diff.append(('modify_nullable', schema, tname, cname, {'existing_type': self.existing_type, 'existing_server_default': self.existing_server_default, 'existing_comment': self.existing_comment}, self.existing_nullable, self.modify_nullable))
        if self.modify_server_default is not False:
            col_diff.append(('modify_default', schema, tname, cname, {'existing_nullable': self.existing_nullable, 'existing_type': self.existing_type, 'existing_comment': self.existing_comment}, self.existing_server_default, self.modify_server_default))
        if self.modify_comment is not False:
            col_diff.append(('modify_comment', schema, tname, cname, {'existing_nullable': self.existing_nullable, 'existing_type': self.existing_type, 'existing_server_default': self.existing_server_default}, self.existing_comment, self.modify_comment))
        return col_diff

    def has_changes(self) -> bool:
        hc1 = self.modify_nullable is not None or self.modify_server_default is not False or self.modify_type is not None or (self.modify_comment is not False)
        if hc1:
            return True
        for kw in self.kw:
            if kw.startswith('modify_'):
                return True
        else:
            return False

    def reverse(self) -> AlterColumnOp:
        kw = self.kw.copy()
        kw['existing_type'] = self.existing_type
        kw['existing_nullable'] = self.existing_nullable
        kw['existing_server_default'] = self.existing_server_default
        kw['existing_comment'] = self.existing_comment
        if self.modify_type is not None:
            kw['modify_type'] = self.modify_type
        if self.modify_nullable is not None:
            kw['modify_nullable'] = self.modify_nullable
        if self.modify_server_default is not False:
            kw['modify_server_default'] = self.modify_server_default
        if self.modify_comment is not False:
            kw['modify_comment'] = self.modify_comment
        all_keys = {m.group(1) for m in [re.match('^(?:existing_|modify_)(.+)$', k) for k in kw] if m}
        for k in all_keys:
            if 'modify_%s' % k in kw:
                swap = kw['existing_%s' % k]
                kw['existing_%s' % k] = kw['modify_%s' % k]
                kw['modify_%s' % k] = swap
        return self.__class__(self.table_name, self.column_name, schema=self.schema, **kw)

    @classmethod
    def alter_column(cls, operations: Operations, table_name: str, column_name: str, *, nullable: Optional[bool]=None, comment: Optional[Union[str, Literal[False]]]=False, server_default: Any=False, new_column_name: Optional[str]=None, type_: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_type: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_server_default: Optional[Union[str, bool, Identity, Computed]]=False, existing_nullable: Optional[bool]=None, existing_comment: Optional[str]=None, schema: Optional[str]=None, **kw: Any) -> None:
        """Issue an "alter column" instruction using the
        current migration context.

        Generally, only that aspect of the column which
        is being changed, i.e. name, type, nullability,
        default, needs to be specified.  Multiple changes
        can also be specified at once and the backend should
        "do the right thing", emitting each change either
        separately or together as the backend allows.

        MySQL has special requirements here, since MySQL
        cannot ALTER a column without a full specification.
        When producing MySQL-compatible migration files,
        it is recommended that the ``existing_type``,
        ``existing_server_default``, and ``existing_nullable``
        parameters be present, if not being altered.

        Type changes which are against the SQLAlchemy
        "schema" types :class:`~sqlalchemy.types.Boolean`
        and  :class:`~sqlalchemy.types.Enum` may also
        add or drop constraints which accompany those
        types on backends that don't support them natively.
        The ``existing_type`` argument is
        used in this case to identify and remove a previous
        constraint that was bound to the type object.

        :param table_name: string name of the target table.
        :param column_name: string name of the target column,
         as it exists before the operation begins.
        :param nullable: Optional; specify ``True`` or ``False``
         to alter the column's nullability.
        :param server_default: Optional; specify a string
         SQL expression, :func:`~sqlalchemy.sql.expression.text`,
         or :class:`~sqlalchemy.schema.DefaultClause` to indicate
         an alteration to the column's default value.
         Set to ``None`` to have the default removed.
        :param comment: optional string text of a new comment to add to the
         column.
        :param new_column_name: Optional; specify a string name here to
         indicate the new name within a column rename operation.
        :param type\\_: Optional; a :class:`~sqlalchemy.types.TypeEngine`
         type object to specify a change to the column's type.
         For SQLAlchemy types that also indicate a constraint (i.e.
         :class:`~sqlalchemy.types.Boolean`, :class:`~sqlalchemy.types.Enum`),
         the constraint is also generated.
        :param autoincrement: set the ``AUTO_INCREMENT`` flag of the column;
         currently understood by the MySQL dialect.
        :param existing_type: Optional; a
         :class:`~sqlalchemy.types.TypeEngine`
         type object to specify the previous type.   This
         is required for all MySQL column alter operations that
         don't otherwise specify a new type, as well as for
         when nullability is being changed on a SQL Server
         column.  It is also used if the type is a so-called
         SQLAlchemy "schema" type which may define a constraint (i.e.
         :class:`~sqlalchemy.types.Boolean`,
         :class:`~sqlalchemy.types.Enum`),
         so that the constraint can be dropped.
        :param existing_server_default: Optional; The existing
         default value of the column.   Required on MySQL if
         an existing default is not being changed; else MySQL
         removes the default.
        :param existing_nullable: Optional; the existing nullability
         of the column.  Required on MySQL if the existing nullability
         is not being changed; else MySQL sets this to NULL.
        :param existing_autoincrement: Optional; the existing autoincrement
         of the column.  Used for MySQL's system of altering a column
         that specifies ``AUTO_INCREMENT``.
        :param existing_comment: string text of the existing comment on the
         column to be maintained.  Required on MySQL if the existing comment
         on the column is not being changed.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.
        :param postgresql_using: String argument which will indicate a
         SQL expression to render within the Postgresql-specific USING clause
         within ALTER COLUMN.    This string is taken directly as raw SQL which
         must explicitly include any necessary quoting or escaping of tokens
         within the expression.

        """
        alt = cls(table_name, column_name, schema=schema, existing_type=existing_type, existing_server_default=existing_server_default, existing_nullable=existing_nullable, existing_comment=existing_comment, modify_name=new_column_name, modify_type=type_, modify_server_default=server_default, modify_nullable=nullable, modify_comment=comment, **kw)
        return operations.invoke(alt)

    @classmethod
    def batch_alter_column(cls, operations: BatchOperations, column_name: str, *, nullable: Optional[bool]=None, comment: Optional[Union[str, Literal[False]]]=False, server_default: Any=False, new_column_name: Optional[str]=None, type_: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_type: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_server_default: Optional[Union[str, bool, Identity, Computed]]=False, existing_nullable: Optional[bool]=None, existing_comment: Optional[str]=None, insert_before: Optional[str]=None, insert_after: Optional[str]=None, **kw: Any) -> None:
        """Issue an "alter column" instruction using the current
        batch migration context.

        Parameters are the same as that of :meth:`.Operations.alter_column`,
        as well as the following option(s):

        :param insert_before: String name of an existing column which this
         column should be placed before, when creating the new table.

        :param insert_after: String name of an existing column which this
         column should be placed after, when creating the new table.  If
         both :paramref:`.BatchOperations.alter_column.insert_before`
         and :paramref:`.BatchOperations.alter_column.insert_after` are
         omitted, the column is inserted after the last existing column
         in the table.

        .. seealso::

            :meth:`.Operations.alter_column`


        """
        alt = cls(operations.impl.table_name, column_name, schema=operations.impl.schema, existing_type=existing_type, existing_server_default=existing_server_default, existing_nullable=existing_nullable, existing_comment=existing_comment, modify_name=new_column_name, modify_type=type_, modify_server_default=server_default, modify_nullable=nullable, modify_comment=comment, insert_before=insert_before, insert_after=insert_after, **kw)
        return operations.invoke(alt)
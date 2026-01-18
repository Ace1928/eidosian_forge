from __future__ import annotations
from contextlib import contextmanager
import re
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List  # noqa
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence  # noqa
from typing import Tuple
from typing import Type  # noqa
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.elements import conv
from . import batch
from . import schemaobj
from .. import util
from ..util import sqla_compat
from ..util.compat import formatannotation_fwdref
from ..util.compat import inspect_formatargspec
from ..util.compat import inspect_getfullargspec
from ..util.sqla_compat import _literal_bindparam
class BatchOperations(AbstractOperations):
    """Modifies the interface :class:`.Operations` for batch mode.

    This basically omits the ``table_name`` and ``schema`` parameters
    from associated methods, as these are a given when running under batch
    mode.

    .. seealso::

        :meth:`.Operations.batch_alter_table`

    Note that as of 0.8, most of the methods on this class are produced
    dynamically using the :meth:`.Operations.register_operation`
    method.

    """
    impl: BatchOperationsImpl

    def _noop(self, operation: Any) -> NoReturn:
        raise NotImplementedError('The %s method does not apply to a batch table alter operation.' % operation)
    if TYPE_CHECKING:

        def add_column(self, column: Column[Any], *, insert_before: Optional[str]=None, insert_after: Optional[str]=None) -> None:
            """Issue an "add column" instruction using the current
            batch migration context.

            .. seealso::

                :meth:`.Operations.add_column`

            """
            ...

        def alter_column(self, column_name: str, *, nullable: Optional[bool]=None, comment: Union[str, Literal[False], None]=False, server_default: Any=False, new_column_name: Optional[str]=None, type_: Union[TypeEngine[Any], Type[TypeEngine[Any]], None]=None, existing_type: Union[TypeEngine[Any], Type[TypeEngine[Any]], None]=None, existing_server_default: Union[str, bool, Identity, Computed, None]=False, existing_nullable: Optional[bool]=None, existing_comment: Optional[str]=None, insert_before: Optional[str]=None, insert_after: Optional[str]=None, **kw: Any) -> None:
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
            ...

        def create_check_constraint(self, constraint_name: str, condition: Union[str, ColumnElement[bool], TextClause], **kw: Any) -> None:
            """Issue a "create check constraint" instruction using the
            current batch migration context.

            The batch form of this call omits the ``source`` and ``schema``
            arguments from the call.

            .. seealso::

                :meth:`.Operations.create_check_constraint`

            """
            ...

        def create_exclude_constraint(self, constraint_name: str, *elements: Any, **kw: Any) -> Optional[Table]:
            """Issue a "create exclude constraint" instruction using the
            current batch migration context.

            .. note::  This method is Postgresql specific, and additionally
               requires at least SQLAlchemy 1.0.

            .. seealso::

                :meth:`.Operations.create_exclude_constraint`

            """
            ...

        def create_foreign_key(self, constraint_name: str, referent_table: str, local_cols: List[str], remote_cols: List[str], *, referent_schema: Optional[str]=None, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, match: Optional[str]=None, **dialect_kw: Any) -> None:
            """Issue a "create foreign key" instruction using the
            current batch migration context.

            The batch form of this call omits the ``source`` and ``source_schema``
            arguments from the call.

            e.g.::

                with batch_alter_table("address") as batch_op:
                    batch_op.create_foreign_key(
                        "fk_user_address",
                        "user",
                        ["user_id"],
                        ["id"],
                    )

            .. seealso::

                :meth:`.Operations.create_foreign_key`

            """
            ...

        def create_index(self, index_name: str, columns: List[str], **kw: Any) -> None:
            """Issue a "create index" instruction using the
            current batch migration context.

            .. seealso::

                :meth:`.Operations.create_index`

            """
            ...

        def create_primary_key(self, constraint_name: str, columns: List[str]) -> None:
            """Issue a "create primary key" instruction using the
            current batch migration context.

            The batch form of this call omits the ``table_name`` and ``schema``
            arguments from the call.

            .. seealso::

                :meth:`.Operations.create_primary_key`

            """
            ...

        def create_table_comment(self, comment: Optional[str], *, existing_comment: Optional[str]=None) -> None:
            """Emit a COMMENT ON operation to set the comment for a table
            using the current batch migration context.

            :param comment: string value of the comment being registered against
             the specified table.
            :param existing_comment: String value of a comment
             already registered on the specified table, used within autogenerate
             so that the operation is reversible, but not required for direct
             use.

            """
            ...

        def create_unique_constraint(self, constraint_name: str, columns: Sequence[str], **kw: Any) -> Any:
            """Issue a "create unique constraint" instruction using the
            current batch migration context.

            The batch form of this call omits the ``source`` and ``schema``
            arguments from the call.

            .. seealso::

                :meth:`.Operations.create_unique_constraint`

            """
            ...

        def drop_column(self, column_name: str, **kw: Any) -> None:
            """Issue a "drop column" instruction using the current
            batch migration context.

            .. seealso::

                :meth:`.Operations.drop_column`

            """
            ...

        def drop_constraint(self, constraint_name: str, type_: Optional[str]=None) -> None:
            """Issue a "drop constraint" instruction using the
            current batch migration context.

            The batch form of this call omits the ``table_name`` and ``schema``
            arguments from the call.

            .. seealso::

                :meth:`.Operations.drop_constraint`

            """
            ...

        def drop_index(self, index_name: str, **kw: Any) -> None:
            """Issue a "drop index" instruction using the
            current batch migration context.

            .. seealso::

                :meth:`.Operations.drop_index`

            """
            ...

        def drop_table_comment(self, *, existing_comment: Optional[str]=None) -> None:
            """Issue a "drop table comment" operation to
            remove an existing comment set on a table using the current
            batch operations context.

            :param existing_comment: An optional string value of a comment already
             registered on the specified table.

            """
            ...

        def execute(self, sqltext: Union[Executable, str], *, execution_options: Optional[dict[str, Any]]=None) -> None:
            """Execute the given SQL using the current migration context.

            .. seealso::

                :meth:`.Operations.execute`

            """
            ...
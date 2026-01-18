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
@Operations.register_operation('execute')
@BatchOperations.register_operation('execute', 'batch_execute')
class ExecuteSQLOp(MigrateOperation):
    """Represent an execute SQL operation."""

    def __init__(self, sqltext: Union[Executable, str], *, execution_options: Optional[dict[str, Any]]=None) -> None:
        self.sqltext = sqltext
        self.execution_options = execution_options

    @classmethod
    def execute(cls, operations: Operations, sqltext: Union[Executable, str], *, execution_options: Optional[dict[str, Any]]=None) -> None:
        """Execute the given SQL using the current migration context.

        The given SQL can be a plain string, e.g.::

            op.execute("INSERT INTO table (foo) VALUES ('some value')")

        Or it can be any kind of Core SQL Expression construct, such as
        below where we use an update construct::

            from sqlalchemy.sql import table, column
            from sqlalchemy import String
            from alembic import op

            account = table("account", column("name", String))
            op.execute(
                account.update()
                .where(account.c.name == op.inline_literal("account 1"))
                .values({"name": op.inline_literal("account 2")})
            )

        Above, we made use of the SQLAlchemy
        :func:`sqlalchemy.sql.expression.table` and
        :func:`sqlalchemy.sql.expression.column` constructs to make a brief,
        ad-hoc table construct just for our UPDATE statement.  A full
        :class:`~sqlalchemy.schema.Table` construct of course works perfectly
        fine as well, though note it's a recommended practice to at least
        ensure the definition of a table is self-contained within the migration
        script, rather than imported from a module that may break compatibility
        with older migrations.

        In a SQL script context, the statement is emitted directly to the
        output stream.   There is *no* return result, however, as this
        function is oriented towards generating a change script
        that can run in "offline" mode.     Additionally, parameterized
        statements are discouraged here, as they *will not work* in offline
        mode.  Above, we use :meth:`.inline_literal` where parameters are
        to be used.

        For full interaction with a connected database where parameters can
        also be used normally, use the "bind" available from the context::

            from alembic import op

            connection = op.get_bind()

            connection.execute(
                account.update()
                .where(account.c.name == "account 1")
                .values({"name": "account 2"})
            )

        Additionally, when passing the statement as a plain string, it is first
        coerced into a :func:`sqlalchemy.sql.expression.text` construct
        before being passed along.  In the less likely case that the
        literal SQL string contains a colon, it must be escaped with a
        backslash, as::

           op.execute(r"INSERT INTO table (foo) VALUES ('\\:colon_value')")


        :param sqltext: Any legal SQLAlchemy expression, including:

        * a string
        * a :func:`sqlalchemy.sql.expression.text` construct.
        * a :func:`sqlalchemy.sql.expression.insert` construct.
        * a :func:`sqlalchemy.sql.expression.update` construct.
        * a :func:`sqlalchemy.sql.expression.delete` construct.
        * Any "executable" described in SQLAlchemy Core documentation,
          noting that no result set is returned.

        .. note::  when passing a plain string, the statement is coerced into
           a :func:`sqlalchemy.sql.expression.text` construct. This construct
           considers symbols with colons, e.g. ``:foo`` to be bound parameters.
           To avoid this, ensure that colon symbols are escaped, e.g.
           ``\\:foo``.

        :param execution_options: Optional dictionary of
         execution options, will be passed to
         :meth:`sqlalchemy.engine.Connection.execution_options`.
        """
        op = cls(sqltext, execution_options=execution_options)
        return operations.invoke(op)

    @classmethod
    def batch_execute(cls, operations: Operations, sqltext: Union[Executable, str], *, execution_options: Optional[dict[str, Any]]=None) -> None:
        """Execute the given SQL using the current migration context.

        .. seealso::

            :meth:`.Operations.execute`

        """
        return cls.execute(operations, sqltext, execution_options=execution_options)

    def to_diff_tuple(self) -> Tuple[str, Union[Executable, str]]:
        return ('execute', self.sqltext)
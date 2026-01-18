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
@Operations.register_operation('bulk_insert')
class BulkInsertOp(MigrateOperation):
    """Represent a bulk insert operation."""

    def __init__(self, table: Union[Table, TableClause], rows: List[Dict[str, Any]], *, multiinsert: bool=True) -> None:
        self.table = table
        self.rows = rows
        self.multiinsert = multiinsert

    @classmethod
    def bulk_insert(cls, operations: Operations, table: Union[Table, TableClause], rows: List[Dict[str, Any]], *, multiinsert: bool=True) -> None:
        """Issue a "bulk insert" operation using the current
        migration context.

        This provides a means of representing an INSERT of multiple rows
        which works equally well in the context of executing on a live
        connection as well as that of generating a SQL script.   In the
        case of a SQL script, the values are rendered inline into the
        statement.

        e.g.::

            from alembic import op
            from datetime import date
            from sqlalchemy.sql import table, column
            from sqlalchemy import String, Integer, Date

            # Create an ad-hoc table to use for the insert statement.
            accounts_table = table(
                "account",
                column("id", Integer),
                column("name", String),
                column("create_date", Date),
            )

            op.bulk_insert(
                accounts_table,
                [
                    {
                        "id": 1,
                        "name": "John Smith",
                        "create_date": date(2010, 10, 5),
                    },
                    {
                        "id": 2,
                        "name": "Ed Williams",
                        "create_date": date(2007, 5, 27),
                    },
                    {
                        "id": 3,
                        "name": "Wendy Jones",
                        "create_date": date(2008, 8, 15),
                    },
                ],
            )

        When using --sql mode, some datatypes may not render inline
        automatically, such as dates and other special types.   When this
        issue is present, :meth:`.Operations.inline_literal` may be used::

            op.bulk_insert(
                accounts_table,
                [
                    {
                        "id": 1,
                        "name": "John Smith",
                        "create_date": op.inline_literal("2010-10-05"),
                    },
                    {
                        "id": 2,
                        "name": "Ed Williams",
                        "create_date": op.inline_literal("2007-05-27"),
                    },
                    {
                        "id": 3,
                        "name": "Wendy Jones",
                        "create_date": op.inline_literal("2008-08-15"),
                    },
                ],
                multiinsert=False,
            )

        When using :meth:`.Operations.inline_literal` in conjunction with
        :meth:`.Operations.bulk_insert`, in order for the statement to work
        in "online" (e.g. non --sql) mode, the
        :paramref:`~.Operations.bulk_insert.multiinsert`
        flag should be set to ``False``, which will have the effect of
        individual INSERT statements being emitted to the database, each
        with a distinct VALUES clause, so that the "inline" values can
        still be rendered, rather than attempting to pass the values
        as bound parameters.

        :param table: a table object which represents the target of the INSERT.

        :param rows: a list of dictionaries indicating rows.

        :param multiinsert: when at its default of True and --sql mode is not
           enabled, the INSERT statement will be executed using
           "executemany()" style, where all elements in the list of
           dictionaries are passed as bound parameters in a single
           list.   Setting this to False results in individual INSERT
           statements being emitted per parameter set, and is needed
           in those cases where non-literal values are present in the
           parameter sets.

        """
        op = cls(table, rows, multiinsert=multiinsert)
        operations.invoke(op)
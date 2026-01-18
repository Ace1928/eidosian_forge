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
@Operations.register_operation('rename_table')
class RenameTableOp(AlterTableOp):
    """Represent a rename table operation."""

    def __init__(self, old_table_name: str, new_table_name: str, *, schema: Optional[str]=None) -> None:
        super().__init__(old_table_name, schema=schema)
        self.new_table_name = new_table_name

    @classmethod
    def rename_table(cls, operations: Operations, old_table_name: str, new_table_name: str, *, schema: Optional[str]=None) -> None:
        """Emit an ALTER TABLE to rename a table.

        :param old_table_name: old name.
        :param new_table_name: new name.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        """
        op = cls(old_table_name, new_table_name, schema=schema)
        return operations.invoke(op)
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
@Operations.register_operation('drop_constraint')
@BatchOperations.register_operation('drop_constraint', 'batch_drop_constraint')
class DropConstraintOp(MigrateOperation):
    """Represent a drop constraint operation."""

    def __init__(self, constraint_name: Optional[sqla_compat._ConstraintNameDefined], table_name: str, type_: Optional[str]=None, *, schema: Optional[str]=None, _reverse: Optional[AddConstraintOp]=None) -> None:
        self.constraint_name = constraint_name
        self.table_name = table_name
        self.constraint_type = type_
        self.schema = schema
        self._reverse = _reverse

    def reverse(self) -> AddConstraintOp:
        return AddConstraintOp.from_constraint(self.to_constraint())

    def to_diff_tuple(self) -> Tuple[str, SchemaItem]:
        if self.constraint_type == 'foreignkey':
            return ('remove_fk', self.to_constraint())
        else:
            return ('remove_constraint', self.to_constraint())

    @classmethod
    def from_constraint(cls, constraint: Constraint) -> DropConstraintOp:
        types = {'unique_constraint': 'unique', 'foreign_key_constraint': 'foreignkey', 'primary_key_constraint': 'primary', 'check_constraint': 'check', 'column_check_constraint': 'check', 'table_or_column_check_constraint': 'check'}
        constraint_table = sqla_compat._table_for_constraint(constraint)
        return cls(sqla_compat.constraint_name_or_none(constraint.name), constraint_table.name, schema=constraint_table.schema, type_=types.get(constraint.__visit_name__), _reverse=AddConstraintOp.from_constraint(constraint))

    def to_constraint(self) -> Constraint:
        if self._reverse is not None:
            constraint = self._reverse.to_constraint()
            constraint.name = self.constraint_name
            constraint_table = sqla_compat._table_for_constraint(constraint)
            constraint_table.name = self.table_name
            constraint_table.schema = self.schema
            return constraint
        else:
            raise ValueError('constraint cannot be produced; original constraint is not present')

    @classmethod
    def drop_constraint(cls, operations: Operations, constraint_name: str, table_name: str, type_: Optional[str]=None, *, schema: Optional[str]=None) -> None:
        """Drop a constraint of the given name, typically via DROP CONSTRAINT.

        :param constraint_name: name of the constraint.
        :param table_name: table name.
        :param type\\_: optional, required on MySQL.  can be
         'foreignkey', 'primary', 'unique', or 'check'.
        :param schema: Optional schema name to operate within.  To control
         quoting of the schema outside of the default behavior, use
         the SQLAlchemy construct
         :class:`~sqlalchemy.sql.elements.quoted_name`.

        """
        op = cls(constraint_name, table_name, type_=type_, schema=schema)
        return operations.invoke(op)

    @classmethod
    def batch_drop_constraint(cls, operations: BatchOperations, constraint_name: str, type_: Optional[str]=None) -> None:
        """Issue a "drop constraint" instruction using the
        current batch migration context.

        The batch form of this call omits the ``table_name`` and ``schema``
        arguments from the call.

        .. seealso::

            :meth:`.Operations.drop_constraint`

        """
        op = cls(constraint_name, operations.impl.table_name, type_=type_, schema=operations.impl.schema)
        return operations.invoke(op)
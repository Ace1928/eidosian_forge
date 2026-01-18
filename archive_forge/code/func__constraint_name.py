from __future__ import annotations
import re
from . import events  # noqa
from .base import _NONE_NAME
from .elements import conv as conv
from .schema import CheckConstraint
from .schema import Column
from .schema import Constraint
from .schema import ForeignKeyConstraint
from .schema import Index
from .schema import PrimaryKeyConstraint
from .schema import Table
from .schema import UniqueConstraint
from .. import event
from .. import exc
@event.listens_for(Constraint, 'after_parent_attach')
@event.listens_for(Index, 'after_parent_attach')
def _constraint_name(const, table):
    if isinstance(table, Column):
        event.listen(table, 'after_parent_attach', lambda col, table: _constraint_name(const, table))
    elif isinstance(table, Table):
        if isinstance(const.name, conv) or const.name is _NONE_NAME:
            return
        newname = _constraint_name_for_table(const, table)
        if newname:
            const.name = newname
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
@event.listens_for(PrimaryKeyConstraint, '_sa_event_column_added_to_pk_constraint')
def _column_added_to_pk_constraint(pk_constraint, col):
    if pk_constraint._implicit_generated:
        table = pk_constraint.table
        pk_constraint.name = None
        newname = _constraint_name_for_table(pk_constraint, table)
        if newname:
            pk_constraint.name = newname
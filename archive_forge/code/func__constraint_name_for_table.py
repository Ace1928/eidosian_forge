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
def _constraint_name_for_table(const, table):
    metadata = table.metadata
    convention = _get_convention(metadata.naming_convention, type(const))
    if isinstance(const.name, conv):
        return const.name
    elif convention is not None and (not isinstance(const.name, conv)) and (const.name is None or 'constraint_name' in convention or const.name is _NONE_NAME):
        return conv(convention % ConventionDict(const, table, metadata.naming_convention))
    elif convention is _NONE_NAME:
        return None
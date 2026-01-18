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
def _key_referred_column_X_name(self, idx):
    fk = self.const.elements[idx]
    return fk.column.name
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
def _key_constraint_name(self):
    if self._const_name in (None, _NONE_NAME):
        raise exc.InvalidRequestError('Naming convention including %(constraint_name)s token requires that constraint is explicitly named.')
    if not isinstance(self._const_name, conv):
        self.const.name = None
    return self._const_name
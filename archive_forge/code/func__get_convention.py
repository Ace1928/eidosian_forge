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
def _get_convention(dict_, key):
    for super_ in key.__mro__:
        if super_ in _prefix_dict and _prefix_dict[super_] in dict_:
            return dict_[_prefix_dict[super_]]
        elif super_ in dict_:
            return dict_[super_]
    else:
        return None
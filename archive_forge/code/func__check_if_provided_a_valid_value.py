from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
@error(INVALID_PROPERTY_VALUE)
def _check_if_provided_a_valid_value(self):
    descriptor = self.obj.lookup(self.attr)
    if descriptor.property.is_valid(self.value):
        return None
    else:
        return f'{self.value!r} is not a valid value for {self.obj}.{self.attr}'
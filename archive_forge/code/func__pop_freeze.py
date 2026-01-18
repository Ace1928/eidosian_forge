from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
def _pop_freeze(self) -> None:
    self._freeze_count -= 1
    if self._freeze_count == 0:
        self.recompute()
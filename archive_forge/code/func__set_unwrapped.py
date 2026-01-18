from __future__ import annotations
import abc
from typing import (
import attr
from ._util import AlreadyUsedError, remove_tb_frames
def _set_unwrapped(self) -> None:
    if self._unwrapped:
        raise AlreadyUsedError
    object.__setattr__(self, '_unwrapped', True)
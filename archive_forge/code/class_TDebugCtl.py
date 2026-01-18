from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TDebugCtl(Protocol):
    """A DebugControl object, or something like it."""

    def should(self, option: str) -> bool:
        """Decide whether to output debug information in category `option`."""

    def write(self, msg: str) -> None:
        """Write a line of debug output."""
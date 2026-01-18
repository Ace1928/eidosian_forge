from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TTraceFn(Protocol):
    """A Python trace function."""

    def __call__(self, frame: FrameType, event: str, arg: Any, lineno: TLineNo | None=None) -> TTraceFn | None:
        ...
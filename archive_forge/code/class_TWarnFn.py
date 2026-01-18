from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TWarnFn(Protocol):
    """A callable warn() function."""

    def __call__(self, msg: str, slug: str | None=None, once: bool=False) -> None:
        ...
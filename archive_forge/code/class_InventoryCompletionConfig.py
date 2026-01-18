from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@dataclasses.dataclass(frozen=True)
class InventoryCompletionConfig(CompletionConfig):
    """Configuration for inventory files."""

    def __init__(self) -> None:
        super().__init__(name='inventory')

    @property
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""
        return False
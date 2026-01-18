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
class PosixSshCompletionConfig(PythonCompletionConfig):
    """Configuration for a POSIX host reachable over SSH."""

    def __init__(self, user: str, host: str) -> None:
        super().__init__(name=f'{user}@{host}', python=','.join(SUPPORTED_PYTHON_VERSIONS))

    @property
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""
        return False
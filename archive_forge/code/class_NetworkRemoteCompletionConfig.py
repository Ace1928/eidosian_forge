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
class NetworkRemoteCompletionConfig(RemoteCompletionConfig):
    """Configuration for remote network platforms."""
    collection: str = ''
    connection: str = ''
    placeholder: bool = False

    def __post_init__(self):
        if not self.placeholder:
            super().__post_init__()
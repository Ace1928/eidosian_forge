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
class WindowsRemoteCompletionConfig(RemoteCompletionConfig):
    """Configuration for remote Windows platforms."""
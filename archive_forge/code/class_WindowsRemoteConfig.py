from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
@dataclasses.dataclass
class WindowsRemoteConfig(RemoteConfig, WindowsConfig):
    """Configuration for a remote Windows host."""

    def get_defaults(self, context: HostContext) -> WindowsRemoteCompletionConfig:
        """Return the default settings."""
        return filter_completion(windows_completion()).get(self.name) or windows_completion().get(self.platform)
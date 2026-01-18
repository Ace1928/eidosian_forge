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
class PosixConfig(HostConfig, metaclass=abc.ABCMeta):
    """Base class for POSIX host configuration."""
    python: t.Optional[PythonConfig] = None

    @property
    @abc.abstractmethod
    def have_root(self) -> bool:
        """True if root is available, otherwise False."""

    @abc.abstractmethod
    def get_defaults(self, context: HostContext) -> PosixCompletionConfig:
        """Return the default settings."""

    def apply_defaults(self, context: HostContext, defaults: CompletionConfig) -> None:
        """Apply default settings."""
        assert isinstance(defaults, PosixCompletionConfig)
        super().apply_defaults(context, defaults)
        self.python = self.python or NativePythonConfig()
        self.python.apply_defaults(context, defaults)
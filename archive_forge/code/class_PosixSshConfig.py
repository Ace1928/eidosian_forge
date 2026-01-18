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
class PosixSshConfig(PosixConfig):
    """Configuration for a POSIX SSH host."""
    user: t.Optional[str] = None
    host: t.Optional[str] = None
    port: t.Optional[int] = None

    def get_defaults(self, context: HostContext) -> PosixSshCompletionConfig:
        """Return the default settings."""
        return PosixSshCompletionConfig(user=self.user, host=self.host)

    @property
    def have_root(self) -> bool:
        """True if root is available, otherwise False."""
        return self.user == 'root'
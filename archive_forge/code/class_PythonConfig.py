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
class PythonConfig(metaclass=abc.ABCMeta):
    """Configuration for Python."""
    version: t.Optional[str] = None
    path: t.Optional[str] = None

    @property
    def tuple(self) -> tuple[int, ...]:
        """Return the Python version as a tuple."""
        return str_to_version(self.version)

    @property
    def major_version(self) -> int:
        """Return the Python major version."""
        return self.tuple[0]

    def apply_defaults(self, context: HostContext, defaults: PosixCompletionConfig) -> None:
        """Apply default settings."""
        if self.version in (None, 'default'):
            self.version = defaults.get_default_python(context.controller)
        if self.path:
            if self.path.endswith('/'):
                self.path = os.path.join(self.path, f'python{self.version}')
        else:
            self.path = defaults.get_python_path(self.version)

    @property
    @abc.abstractmethod
    def is_managed(self) -> bool:
        """
        True if this Python is a managed instance, otherwise False.
        Managed instances are used exclusively by ansible-test and can safely have requirements installed without explicit permission from the user.
        """
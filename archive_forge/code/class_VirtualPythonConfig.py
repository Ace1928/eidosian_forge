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
class VirtualPythonConfig(PythonConfig):
    """Configuration for Python in a virtual environment."""
    system_site_packages: t.Optional[bool] = None

    def apply_defaults(self, context: HostContext, defaults: PosixCompletionConfig) -> None:
        """Apply default settings."""
        super().apply_defaults(context, defaults)
        if self.system_site_packages is None:
            self.system_site_packages = False

    @property
    def is_managed(self) -> bool:
        """
        True if this Python is a managed instance, otherwise False.
        Managed instances are used exclusively by ansible-test and can safely have requirements installed without explicit permission from the user.
        """
        return True
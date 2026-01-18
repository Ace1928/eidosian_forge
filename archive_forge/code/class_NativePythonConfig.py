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
class NativePythonConfig(PythonConfig):
    """Configuration for native Python."""

    @property
    def is_managed(self) -> bool:
        """
        True if this Python is a managed instance, otherwise False.
        Managed instances are used exclusively by ansible-test and can safely have requirements installed without explicit permission from the user.
        """
        return False
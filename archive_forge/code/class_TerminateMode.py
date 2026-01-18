from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
class TerminateMode(enum.Enum):
    """When to terminate instances."""
    ALWAYS = enum.auto()
    NEVER = enum.auto()
    SUCCESS = enum.auto()

    def __str__(self):
        return self.name.lower()
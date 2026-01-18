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
@dataclasses.dataclass(frozen=True)
class ModulesConfig:
    """Configuration for modules."""
    python_requires: str
    python_versions: tuple[str, ...]
    controller_only: bool
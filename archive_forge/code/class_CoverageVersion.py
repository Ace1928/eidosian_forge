from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
@dataclasses.dataclass(frozen=True)
class CoverageVersion:
    """Details about a coverage version and its supported Python versions."""
    coverage_version: str
    schema_version: int
    min_python: tuple[int, int]
    max_python: tuple[int, int]
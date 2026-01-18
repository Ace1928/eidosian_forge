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
class CoverageError(ApplicationError):
    """Exception caused while attempting to read a coverage file."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f'Error reading coverage file "{os.path.relpath(path)}": {message}')
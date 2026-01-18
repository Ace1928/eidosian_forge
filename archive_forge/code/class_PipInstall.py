from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
@dataclasses.dataclass(frozen=True)
class PipInstall(PipCommand):
    """Details required to perform a pip install."""
    requirements: list[tuple[str, str]]
    constraints: list[tuple[str, str]]
    packages: list[str]

    def has_package(self, name: str) -> bool:
        """Return True if the specified package will be installed, otherwise False."""
        name = name.lower()
        return any((name in package.lower() for package in self.packages)) or any((name in contents.lower() for path, contents in self.requirements))
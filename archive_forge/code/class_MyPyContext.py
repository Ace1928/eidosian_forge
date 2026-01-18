from __future__ import annotations
import dataclasses
import os
import re
import typing as t
from . import (
from ...constants import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...host_configs import (
@dataclasses.dataclass(frozen=True)
class MyPyContext:
    """Context details for a single run of mypy."""
    name: str
    paths: list[str]
    python_versions: tuple[str, ...]
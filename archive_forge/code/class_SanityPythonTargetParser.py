from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class SanityPythonTargetParser(PythonTargetParser):
    """Composite argument parser for a sanity Python target."""

    def __init__(self) -> None:
        super().__init__(allow_venv=False)
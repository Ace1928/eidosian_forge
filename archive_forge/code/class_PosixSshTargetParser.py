from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class PosixSshTargetParser(PosixTargetParser):
    """Composite argument parser for a POSIX SSH target."""

    @property
    def option_name(self) -> str:
        """The option name used for this parser."""
        return '--target-posix'
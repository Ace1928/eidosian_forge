from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class NetworkSshTargetParser(NetworkTargetParser):
    """Composite argument parser for a network SSH target."""

    @property
    def option_name(self) -> str:
        """The option name used for this parser."""
        return '--target-network'

    @property
    def allow_inventory(self) -> bool:
        """True if inventory is allowed, otherwise False."""
        return False

    @property
    def limit_one(self) -> bool:
        """True if only one target is allowed, otherwise False."""
        return True
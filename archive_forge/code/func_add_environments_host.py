from __future__ import annotations
import argparse
import enum
import functools
import typing as t
from ..constants import (
from ..util import (
from ..completion import (
from ..cli.argparsing import (
from ..cli.argparsing.actions import (
from ..cli.actions import (
from ..cli.compat import (
from ..config import (
from .completers import (
from .converters import (
from .epilog import (
from ..ci import (
def add_environments_host(environments_parser: argparse.ArgumentParser, controller_mode: ControllerMode, target_mode: TargetMode) -> None:
    """Add environment arguments for the given host and argument modes."""
    environments_exclusive_group: argparse.ArgumentParser = environments_parser.add_mutually_exclusive_group()
    add_environment_local(environments_exclusive_group)
    add_environment_venv(environments_exclusive_group, environments_parser)
    if controller_mode == ControllerMode.DELEGATED:
        add_environment_remote(environments_exclusive_group, environments_parser, target_mode)
        add_environment_docker(environments_exclusive_group, environments_parser, target_mode)
    if target_mode == TargetMode.WINDOWS_INTEGRATION:
        add_environment_windows(environments_parser)
    if target_mode == TargetMode.NETWORK_INTEGRATION:
        add_environment_network(environments_parser)
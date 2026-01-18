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
def add_environment_network(environments_parser: argparse.ArgumentParser) -> None:
    """Add environment arguments for running on a windows host."""
    register_completer(environments_parser.add_argument('--platform', metavar='PLATFORM', action='append', help='network platform/version'), complete_network_platform)
    register_completer(environments_parser.add_argument('--platform-collection', type=key_value_type, metavar='PLATFORM=COLLECTION', action='append', help='collection used to test platform'), complete_network_platform_collection)
    register_completer(environments_parser.add_argument('--platform-connection', type=key_value_type, metavar='PLATFORM=CONNECTION', action='append', help='connection used to test platform'), complete_network_platform_connection)
    environments_parser.add_argument('--inventory', metavar='PATH', help='path to inventory used for tests')
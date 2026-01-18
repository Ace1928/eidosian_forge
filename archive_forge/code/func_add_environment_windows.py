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
def add_environment_windows(environments_parser: argparse.ArgumentParser) -> None:
    """Add environment arguments for running on a windows host."""
    register_completer(environments_parser.add_argument('--windows', metavar='VERSION', action='append', help='windows version'), complete_windows)
    environments_parser.add_argument('--inventory', metavar='PATH', help='path to inventory used for tests')
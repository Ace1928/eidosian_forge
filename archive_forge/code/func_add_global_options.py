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
def add_global_options(parser: argparse.ArgumentParser, controller_mode: ControllerMode):
    """Add global options for controlling the test environment that work with both the legacy and composite options."""
    global_parser = t.cast(argparse.ArgumentParser, parser.add_argument_group(title='global environment arguments'))
    global_parser.add_argument('--containers', metavar='JSON', help=argparse.SUPPRESS)
    global_parser.add_argument('--pypi-proxy', action='store_true', help=argparse.SUPPRESS)
    global_parser.add_argument('--pypi-endpoint', metavar='URI', help=argparse.SUPPRESS)
    global_parser.add_argument('--requirements', action='store_true', default=False, help='install command requirements')
    add_global_remote(global_parser, controller_mode)
    add_global_docker(global_parser, controller_mode)
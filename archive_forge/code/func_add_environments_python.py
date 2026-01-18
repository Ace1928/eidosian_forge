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
def add_environments_python(environments_parser: argparse.ArgumentParser, target_mode: TargetMode) -> None:
    """Add environment arguments to control the Python version(s) used."""
    python_versions: tuple[str, ...]
    if target_mode.has_python:
        python_versions = SUPPORTED_PYTHON_VERSIONS
    else:
        python_versions = CONTROLLER_PYTHON_VERSIONS
    environments_parser.add_argument('--python', metavar='X.Y', choices=python_versions + ('default',), help='python version: %s' % ', '.join(python_versions))
    environments_parser.add_argument('--python-interpreter', metavar='PATH', help='path to the python interpreter')
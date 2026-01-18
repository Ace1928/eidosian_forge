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
class ControllerMode(enum.Enum):
    """Type of provisioning to use for the controller."""
    NO_DELEGATION = enum.auto()
    ORIGIN = enum.auto()
    DELEGATED = enum.auto()
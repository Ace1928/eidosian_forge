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
def add_environment_remote(exclusive_parser: argparse.ArgumentParser, environments_parser: argparse.ArgumentParser, target_mode: TargetMode) -> None:
    """Add environment arguments for running in ansible-core-ci provisioned remote virtual machines."""
    if target_mode == TargetMode.POSIX_INTEGRATION:
        remote_platforms = get_remote_platform_choices()
    elif target_mode == TargetMode.SHELL:
        remote_platforms = sorted(set(get_remote_platform_choices()) | set(get_windows_platform_choices()))
    else:
        remote_platforms = get_remote_platform_choices(True)
    suppress = None if get_ci_provider().supports_core_ci_auth() else argparse.SUPPRESS
    register_completer(exclusive_parser.add_argument('--remote', metavar='NAME', help=suppress or 'run from a remote instance'), functools.partial(complete_choices, remote_platforms))
    environments_parser.add_argument('--remote-provider', metavar='PR', choices=REMOTE_PROVIDERS, help=suppress or 'remote provider to use: %(choices)s')
    environments_parser.add_argument('--remote-arch', metavar='ARCH', choices=REMOTE_ARCHITECTURES, help=suppress or 'remote arch to use: %(choices)s')
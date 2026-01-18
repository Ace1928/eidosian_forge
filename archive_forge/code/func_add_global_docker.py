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
def add_global_docker(parser: argparse.ArgumentParser, controller_mode: ControllerMode) -> None:
    """Add global options for Docker."""
    if controller_mode != ControllerMode.DELEGATED:
        parser.set_defaults(docker_network=None, docker_terminate=None, prime_containers=False, dev_systemd_debug=False, dev_probe_cgroups=None)
        return
    parser.add_argument('--docker-network', metavar='NET', help='run using the specified network')
    parser.add_argument('--docker-terminate', metavar='T', default=TerminateMode.ALWAYS, type=TerminateMode, action=EnumAction, help='terminate the container: %(choices)s (default: %(default)s)')
    parser.add_argument('--prime-containers', action='store_true', help='download containers without running tests')
    suppress = None if get_ci_provider().supports_core_ci_auth() else argparse.SUPPRESS
    parser.add_argument('--dev-systemd-debug', action='store_true', help=suppress or 'enable systemd debugging in containers')
    parser.add_argument('--dev-probe-cgroups', metavar='DIR', nargs='?', const='', help=suppress or 'probe container cgroups, with optional log dir')
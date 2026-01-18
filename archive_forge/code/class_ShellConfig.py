from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
class ShellConfig(EnvironmentConfig):
    """Configuration for the shell command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args, 'shell')
        self.cmd: list[str] = args.cmd
        self.raw: bool = args.raw
        self.check_layout = self.delegate
        self.interactive = sys.stdin.isatty() and (not args.cmd)
        self.export: t.Optional[str] = args.export
        self.display_stderr = True
from __future__ import annotations
import datetime
import os
import platform
import sys
import typing as t
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...docker_util import (
from ...constants import (
from ...ci import (
from ...timeout import (
class EnvConfig(CommonConfig):
    """Configuration for the `env` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args, 'env')
        self.show: bool = args.show
        self.dump: bool = args.dump
        self.timeout: int | float | None = args.timeout
        self.list_files: bool = args.list_files
        if not self.show and (not self.dump) and (self.timeout is None) and (not self.list_files):
            self.show = True
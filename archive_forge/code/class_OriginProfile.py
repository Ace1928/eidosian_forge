from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
class OriginProfile(ControllerHostProfile[OriginConfig]):
    """Host profile for origin."""

    def get_origin_controller_connection(self) -> LocalConnection:
        """Return a connection for accessing the host as a controller from the origin."""
        return LocalConnection(self.args)

    def get_working_directory(self) -> str:
        """Return the working directory for the host."""
        return os.getcwd()
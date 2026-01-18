from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
class SupportContainerContext:
    """Context object for tracking information relating to access of support containers."""

    def __init__(self, containers: ContainerDatabase, process: t.Optional[SshProcess]) -> None:
        self.containers = containers
        self.process = process

    def close(self) -> None:
        """Close the process maintaining the port forwards."""
        if not self.process:
            return
        self.process.terminate()
        display.info('Waiting for the session SSH port forwarding process to terminate.', verbosity=1)
        self.process.wait()
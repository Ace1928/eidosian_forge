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
class ControlGroupError(ApplicationError):
    """Raised when the container host does not have the necessary cgroup support to run a container."""

    def __init__(self, args: CommonConfig, reason: str) -> None:
        engine = require_docker().command
        dd_wsl2 = get_docker_info(args).docker_desktop_wsl2
        message = f'\n{reason}\n\nRun the following commands as root on the container host to resolve this issue:\n\n  mkdir /sys/fs/cgroup/systemd\n  mount cgroup -t cgroup /sys/fs/cgroup/systemd -o none,name=systemd,xattr\n  chown -R {{user}}:{{group}} /sys/fs/cgroup/systemd  # only when rootless\n\nNOTE: These changes must be applied each time the container host is rebooted.\n'.strip()
        podman_message = "\n      If rootless Podman is already running [1], you may need to stop it before\n      containers are able to use the new mount point.\n\n[1] Check for 'podman' and 'catatonit' processes.\n"
        dd_wsl_message = f'\n      When using Docker Desktop with WSL2, additional configuration [1] is required.\n\n[1] {get_docs_url('https://docs.ansible.com/ansible-core/devel/dev_guide/testing_running_locally.html#docker-desktop-with-wsl2')}\n'
        if engine == 'podman':
            message += podman_message
        elif dd_wsl2:
            message += dd_wsl_message
        message = message.strip()
        super().__init__(message)
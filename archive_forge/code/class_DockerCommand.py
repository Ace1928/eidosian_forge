from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
class DockerCommand:
    """Details about the available docker command."""

    def __init__(self, command: str, executable: str, version: str) -> None:
        self.command = command
        self.executable = executable
        self.version = version

    @staticmethod
    def detect() -> t.Optional[DockerCommand]:
        """Detect and return the available docker command, or None."""
        if os.environ.get('ANSIBLE_TEST_PREFER_PODMAN'):
            commands = list(reversed(DOCKER_COMMANDS))
        else:
            commands = DOCKER_COMMANDS
        for command in commands:
            executable = find_executable(command, required=False)
            if executable:
                version = raw_command([command, '-v'], env=docker_environment(), capture=True)[0].strip()
                if command == 'docker' and 'podman' in version:
                    continue
                display.info('Detected "%s" container runtime version: %s' % (command, version), verbosity=1)
                return DockerCommand(command, executable, version)
        return None
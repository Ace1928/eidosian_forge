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
class ContainerAccess:
    """Information needed for one test host to access a single container supporting tests."""

    def __init__(self, host_ip: str, names: list[str], ports: t.Optional[list[int]], forwards: t.Optional[dict[int, int]]) -> None:
        self.host_ip = host_ip
        self.names = names
        self.ports = ports
        self.forwards = forwards

    def port_map(self) -> list[tuple[int, int]]:
        """Return a port map for accessing this container."""
        if self.forwards:
            ports = list(self.forwards.items())
        else:
            ports = [(port, port) for port in self.ports]
        return ports

    @staticmethod
    def from_dict(data: dict[str, t.Any]) -> ContainerAccess:
        """Return a ContainerAccess instance from the given dict."""
        forwards = data.get('forwards')
        if forwards:
            forwards = dict(((int(key), value) for key, value in forwards.items()))
        return ContainerAccess(host_ip=data['host_ip'], names=data['names'], ports=data.get('ports'), forwards=forwards)

    def to_dict(self) -> dict[str, t.Any]:
        """Return a dict of the current instance."""
        value: dict[str, t.Any] = dict(host_ip=self.host_ip, names=self.names)
        if self.ports:
            value.update(ports=self.ports)
        if self.forwards:
            value.update(forwards=self.forwards)
        return value
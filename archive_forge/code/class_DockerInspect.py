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
class DockerInspect:
    """The results of `docker inspect` for a single container."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        self.args = args
        self.inspection = inspection

    @property
    def id(self) -> str:
        """Return the ID of the container."""
        return self.inspection['Id']

    @property
    def network_settings(self) -> dict[str, t.Any]:
        """Return a dictionary of the container network settings."""
        return self.inspection['NetworkSettings']

    @property
    def state(self) -> dict[str, t.Any]:
        """Return a dictionary of the container state."""
        return self.inspection['State']

    @property
    def config(self) -> dict[str, t.Any]:
        """Return a dictionary of the container configuration."""
        return self.inspection['Config']

    @property
    def ports(self) -> dict[str, list[dict[str, str]]]:
        """Return a dictionary of ports the container has published."""
        return self.network_settings['Ports']

    @property
    def networks(self) -> t.Optional[dict[str, dict[str, t.Any]]]:
        """Return a dictionary of the networks the container is attached to, or None if running under podman, which does not support networks."""
        return self.network_settings.get('Networks')

    @property
    def running(self) -> bool:
        """Return True if the container is running, otherwise False."""
        return self.state['Running']

    @property
    def pid(self) -> int:
        """Return the PID of the init process."""
        if self.args.explain:
            return 0
        return self.state['Pid']

    @property
    def env(self) -> list[str]:
        """Return a list of the environment variables used to create the container."""
        return self.config['Env']

    @property
    def image(self) -> str:
        """Return the image used to create the container."""
        return self.config['Image']

    def env_dict(self) -> dict[str, str]:
        """Return a dictionary of the environment variables used to create the container."""
        return dict(((item[0], item[1]) for item in [e.split('=', 1) for e in self.env]))

    def get_tcp_port(self, port: int) -> t.Optional[list[dict[str, str]]]:
        """Return a list of the endpoints published by the container for the specified TCP port, or None if it is not published."""
        return self.ports.get('%d/tcp' % port)

    def get_network_names(self) -> t.Optional[list[str]]:
        """Return a list of the network names the container is attached to."""
        if self.networks is None:
            return None
        return sorted(self.networks)

    def get_network_name(self) -> str:
        """Return the network name the container is attached to. Raises an exception if no network, or more than one, is attached."""
        networks = self.get_network_names()
        if not networks:
            raise ApplicationError('No network found for Docker container: %s.' % self.id)
        if len(networks) > 1:
            raise ApplicationError('Found multiple networks for Docker container %s instead of only one: %s' % (self.id, ', '.join(networks)))
        return networks[0]
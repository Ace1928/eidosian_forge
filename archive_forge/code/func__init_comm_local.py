import logging
import socket
from functools import lru_cache
from typing import Optional
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.environments.lightning import find_free_network_port
def _init_comm_local(self) -> None:
    hostname = socket.gethostname()
    all_hostnames = self._comm_world.gather(hostname, root=0)
    unique_hosts = sorted(set(all_hostnames)) if all_hostnames is not None else []
    unique_hosts = self._comm_world.bcast(unique_hosts, root=0)
    self._node_rank = unique_hosts.index(hostname)
    self._comm_local = self._comm_world.Split(color=self._node_rank)
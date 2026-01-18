import logging
import os
import socket
from typing import Dict, List
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.cloud_io import get_filesystem
def _get_node_rank(self) -> int:
    """A helper method for getting the node rank.

        The node rank is determined by the position of the current node in the list of hosts used in the job. This is
        calculated by reading all hosts from ``LSB_DJOB_RANKFILE`` and finding this node's hostname in the list.

        """
    hosts = self._read_hosts()
    count: Dict[str, int] = {}
    for host in hosts:
        if host not in count:
            count[host] = len(count)
    return count[socket.gethostname()]
import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
class OnPremCoordinatorState(ClusterState):
    """Generates & updates the state file of CoordinatorSenderNodeProvider.

    Unlike ClusterState, which generates a cluster specific file with
    predefined head and worker ips, OnPremCoordinatorState overwrites
    ClusterState's __init__ function to generate and manage a unified
    file of the status of all the nodes for multiple clusters.
    """

    def __init__(self, lock_path, save_path, list_of_node_ips):
        self.lock = RLock()
        self.file_lock = FileLock(lock_path)
        self.save_path = save_path
        with self.lock:
            with self.file_lock:
                if os.path.exists(self.save_path):
                    nodes = json.loads(open(self.save_path).read())
                else:
                    nodes = {}
                logger.info('OnPremCoordinatorState: Loaded on prem coordinator state: {}'.format(nodes))
                for node_ip in list(nodes):
                    if node_ip not in list_of_node_ips:
                        del nodes[node_ip]
                for node_ip in list_of_node_ips:
                    if node_ip not in nodes:
                        nodes[node_ip] = {'tags': {}, 'state': 'terminated'}
                assert len(nodes) == len(list_of_node_ips)
                with open(self.save_path, 'w') as f:
                    logger.info('OnPremCoordinatorState: Writing on prem coordinator state: {}'.format(nodes))
                    f.write(json.dumps(nodes))
import json
import logging
import os
import socket
from threading import RLock
from filelock import FileLock
from ray.autoscaler._private.local.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
class LocalNodeProvider(NodeProvider):
    """NodeProvider for private/local clusters.

    `node_id` is overloaded to also be `node_ip` in this class.

    When `cluster_name` is provided, it manages a single cluster in a cluster
    specific state file. But when `cluster_name` is None, it manages multiple
    clusters in a unified state file that requires each node to be tagged with
    TAG_RAY_CLUSTER_NAME in create and non_terminated_nodes function calls to
    associate each node with the right cluster.

    The current use case of managing multiple clusters is by
    OnPremCoordinatorServer which receives node provider HTTP requests
    from CoordinatorSenderNodeProvider and uses LocalNodeProvider to get
    the responses.
    """

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        if cluster_name:
            lock_path = get_lock_path(cluster_name)
            state_path = get_state_path(cluster_name)
            self.state = ClusterState(lock_path, state_path, provider_config)
            self.use_coordinator = False
        else:
            self.state = OnPremCoordinatorState('/tmp/coordinator.lock', '/tmp/coordinator.state', provider_config['list_of_node_ips'])
            self.use_coordinator = True

    def non_terminated_nodes(self, tag_filters):
        workers = self.state.get()
        matching_ips = []
        for worker_ip, info in workers.items():
            if info['state'] == 'terminated':
                continue
            ok = True
            for k, v in tag_filters.items():
                if info['tags'].get(k) != v:
                    ok = False
                    break
            if ok:
                matching_ips.append(worker_ip)
        return matching_ips

    def is_running(self, node_id):
        return self.state.get()[node_id]['state'] == 'running'

    def is_terminated(self, node_id):
        return not self.is_running(node_id)

    def node_tags(self, node_id):
        return self.state.get()[node_id]['tags']

    def external_ip(self, node_id):
        """Returns an external ip if the user has supplied one.
        Otherwise, use the same logic as internal_ip below.

        This can be used to call ray up from outside the network, for example
        if the Ray cluster exists in an AWS VPC and we're interacting with
        the cluster from a laptop (where using an internal_ip will not work).

        Useful for debugging the local node provider with cloud VMs."""
        node_state = self.state.get()[node_id]
        ext_ip = node_state.get('external_ip')
        if ext_ip:
            return ext_ip
        else:
            return socket.gethostbyname(node_id)

    def internal_ip(self, node_id):
        return socket.gethostbyname(node_id)

    def set_node_tags(self, node_id, tags):
        with self.state.file_lock:
            info = self.state.get()[node_id]
            info['tags'].update(tags)
            self.state.put(node_id, info)

    def create_node(self, node_config, tags, count):
        """Creates min(count, currently available) nodes."""
        node_type = tags[TAG_RAY_NODE_KIND]
        with self.state.file_lock:
            workers = self.state.get()
            for node_id, info in workers.items():
                if info['state'] == 'terminated' and (self.use_coordinator or info['tags'][TAG_RAY_NODE_KIND] == node_type):
                    info['tags'] = tags
                    info['state'] = 'running'
                    self.state.put(node_id, info)
                    count = count - 1
                    if count == 0:
                        return

    def terminate_node(self, node_id):
        workers = self.state.get()
        info = workers[node_id]
        info['state'] = 'terminated'
        self.state.put(node_id, info)

    @staticmethod
    def bootstrap_config(cluster_config):
        return bootstrap_local(cluster_config)
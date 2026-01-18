import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
class FakeMultiNodeProvider(NodeProvider):
    """A node provider that implements multi-node on a single machine.

    This is used for laptop mode testing of autoscaling functionality."""

    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.lock = RLock()
        if 'RAY_FAKE_CLUSTER' not in os.environ:
            raise RuntimeError('FakeMultiNodeProvider requires ray to be started with RAY_FAKE_CLUSTER=1 ray start ...')
        self._nodes = {FAKE_HEAD_NODE_ID: {'tags': {TAG_RAY_NODE_KIND: NODE_KIND_HEAD, TAG_RAY_USER_NODE_TYPE: FAKE_HEAD_NODE_TYPE, TAG_RAY_NODE_NAME: FAKE_HEAD_NODE_ID, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}}}
        self._next_node_id = 0

    def _next_hex_node_id(self):
        self._next_node_id += 1
        base = 'fffffffffffffffffffffffffffffffffffffffffffffffffff'
        return base + str(self._next_node_id).zfill(5)

    def non_terminated_nodes(self, tag_filters):
        with self.lock:
            nodes = []
            for node_id in self._nodes:
                tags = self.node_tags(node_id)
                ok = True
                for k, v in tag_filters.items():
                    if tags.get(k) != v:
                        ok = False
                if ok:
                    nodes.append(node_id)
            return nodes

    def is_running(self, node_id):
        with self.lock:
            return node_id in self._nodes

    def is_terminated(self, node_id):
        with self.lock:
            return node_id not in self._nodes

    def node_tags(self, node_id):
        with self.lock:
            return self._nodes[node_id]['tags']

    def _get_ip(self, node_id: str) -> Optional[str]:
        return node_id

    def external_ip(self, node_id):
        return self._get_ip(node_id)

    def internal_ip(self, node_id):
        return self._get_ip(node_id)

    def set_node_tags(self, node_id, tags):
        raise AssertionError('Readonly node provider cannot be updated')

    def create_node_with_resources_and_labels(self, node_config, tags, count, resources, labels):
        with self.lock:
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            next_id = self._next_hex_node_id()
            ray_params = ray._private.parameter.RayParams(min_worker_port=0, max_worker_port=0, dashboard_port=None, num_cpus=resources.pop('CPU', 0), num_gpus=resources.pop('GPU', 0), object_store_memory=resources.pop('object_store_memory', None), resources=resources, labels=labels, redis_address='{}:6379'.format(ray._private.services.get_node_ip_address()), gcs_address='{}:6379'.format(ray._private.services.get_node_ip_address()), env_vars={'RAY_OVERRIDE_NODE_ID_FOR_TESTING': next_id, ray_constants.RESOURCES_ENVIRONMENT_VARIABLE: json.dumps(resources), ray_constants.LABELS_ENVIRONMENT_VARIABLE: json.dumps(labels)})
            node = ray._private.node.Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False)
            self._nodes[next_id] = {'tags': {TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_USER_NODE_TYPE: node_type, TAG_RAY_NODE_NAME: next_id, TAG_RAY_NODE_STATUS: STATUS_UP_TO_DATE}, 'node': node}

    def terminate_node(self, node_id):
        with self.lock:
            try:
                node = self._nodes.pop(node_id)
            except Exception as e:
                raise e
            self._terminate_node(node)

    def _terminate_node(self, node):
        node['node'].kill_all_processes(check_alive=False, allow_graceful=True)

    @staticmethod
    def bootstrap_config(cluster_config):
        return cluster_config
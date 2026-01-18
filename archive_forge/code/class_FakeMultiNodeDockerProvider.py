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
class FakeMultiNodeDockerProvider(FakeMultiNodeProvider):
    """A node provider that implements multi-node on a single machine.

    This is used for laptop mode testing of multi node functionality
    where each node has their own FS and IP."""

    def __init__(self, provider_config, cluster_name):
        super(FakeMultiNodeDockerProvider, self).__init__(provider_config, cluster_name)
        fake_head = copy.deepcopy(self._nodes)
        self._project_name = self.provider_config['project_name']
        self._docker_image = self.provider_config['image']
        self._host_gcs_port = self.provider_config.get('host_gcs_port', FAKE_DOCKER_DEFAULT_GCS_PORT)
        self._host_object_manager_port = self.provider_config.get('host_object_manager_port', FAKE_DOCKER_DEFAULT_OBJECT_MANAGER_PORT)
        self._host_client_port = self.provider_config.get('host_client_port', FAKE_DOCKER_DEFAULT_CLIENT_PORT)
        self._head_resources = self.provider_config['head_resources']
        self._volume_dir = self.provider_config['shared_volume_dir']
        self._mounted_cluster_dir = os.path.join(self._volume_dir, 'shared')
        if not self.in_docker_container:
            os.makedirs(self._mounted_cluster_dir, mode=493, exist_ok=True)
        self._boostrap_config_path = os.path.join(self._volume_dir, 'bootstrap_config.yaml')
        self._private_key_path = os.path.join(self._volume_dir, 'bootstrap_key.pem')
        self._public_key_path = os.path.join(self._volume_dir, 'bootstrap_key.pem.pub')
        if not self.in_docker_container:
            if not os.path.exists(self._private_key_path):
                subprocess.check_output(f'ssh-keygen -b 2048 -t rsa -q -N "" -f {self._private_key_path}', shell=True)
            if not os.path.exists(self._public_key_path):
                subprocess.check_output(f'ssh-keygen -y -f {self._private_key_path} > {self._public_key_path}', shell=True)
        self._docker_compose_config_path = os.path.join(self._volume_dir, 'docker-compose.yaml')
        self._docker_compose_config = None
        self._node_state_path = os.path.join(self._volume_dir, 'nodes.json')
        self._docker_status_path = os.path.join(self._volume_dir, 'status.json')
        self._load_node_state()
        if FAKE_HEAD_NODE_ID not in self._nodes:
            self._nodes = copy.deepcopy(fake_head)
        self._nodes[FAKE_HEAD_NODE_ID]['node_spec'] = self._create_node_spec_with_resources(head=True, node_id=FAKE_HEAD_NODE_ID, resources=self._head_resources)
        self._possibly_terminated_nodes = dict()
        self._cleanup_interval = provider_config.get('cleanup_interval', 9.5)
        self._docker_status = {}
        self._update_docker_compose_config()
        self._update_docker_status()
        self._save_node_state()

    @property
    def in_docker_container(self):
        return os.path.exists(os.path.join(self._volume_dir, '.in_docker'))

    def _create_node_spec_with_resources(self, head: bool, node_id: str, resources: Dict[str, Any]):
        resources = resources.copy()
        node_dir = os.path.join(self._volume_dir, 'nodes', node_id)
        os.makedirs(node_dir, mode=511, exist_ok=True)
        resource_str = json.dumps(resources, indent=None)
        return create_node_spec(head=head, docker_image=self._docker_image, mounted_cluster_dir=self._mounted_cluster_dir, mounted_node_dir=node_dir, num_cpus=resources.pop('CPU', 0), num_gpus=resources.pop('GPU', 0), host_gcs_port=self._host_gcs_port, host_object_manager_port=self._host_object_manager_port, host_client_port=self._host_client_port, resources=resources, env_vars={'RAY_OVERRIDE_NODE_ID_FOR_TESTING': node_id, ray_constants.RESOURCES_ENVIRONMENT_VARIABLE: resource_str, **self.provider_config.get('env_vars', {})}, volume_dir=self._volume_dir, node_state_path=self._node_state_path, docker_status_path=self._docker_status_path, docker_compose_path=self._docker_compose_config_path, bootstrap_config_path=self._boostrap_config_path, public_key_path=self._public_key_path, private_key_path=self._private_key_path)

    def _load_node_state(self) -> bool:
        if not os.path.exists(self._node_state_path):
            return False
        try:
            with open(self._node_state_path, 'rt') as f:
                nodes = json.load(f)
        except Exception:
            return False
        if not nodes:
            return False
        self._nodes = nodes
        return True

    def _save_node_state(self):
        with open(self._node_state_path, 'wt') as f:
            json.dump(self._nodes, f)
        if not self.in_docker_container:
            os.chmod(self._node_state_path, 511)

    def _update_docker_compose_config(self):
        config = copy.deepcopy(DOCKER_COMPOSE_SKELETON)
        config['services'] = {}
        for node_id, node in self._nodes.items():
            config['services'][node_id] = node['node_spec']
        with open(self._docker_compose_config_path, 'wt') as f:
            yaml.safe_dump(config, f)

    def _update_docker_status(self):
        if not os.path.exists(self._docker_status_path):
            return
        with open(self._docker_status_path, 'rt') as f:
            self._docker_status = json.load(f)

    def _update_nodes(self):
        for node_id in list(self._nodes):
            if not self._is_docker_running(node_id):
                self._possibly_terminated_nodes.setdefault(node_id, time.monotonic())
            else:
                self._possibly_terminated_nodes.pop(node_id, None)
        self._cleanup_nodes()

    def _cleanup_nodes(self):
        for node_id, timestamp in list(self._possibly_terminated_nodes.items()):
            if time.monotonic() > timestamp + self._cleanup_interval:
                if not self._is_docker_running(node_id):
                    self._nodes.pop(node_id, None)
                self._possibly_terminated_nodes.pop(node_id, None)
        self._save_node_state()

    def _container_name(self, node_id):
        node_status = self._docker_status.get(node_id, {})
        timeout = time.monotonic() + 60
        while not node_status:
            if time.monotonic() > timeout:
                raise RuntimeError(f'Container for {node_id} never became available.')
            time.sleep(1)
            self._update_docker_status()
            node_status = self._docker_status.get(node_id, {})
        return node_status['Name']

    def _is_docker_running(self, node_id):
        self._update_docker_status()
        return self._docker_status.get(node_id, {}).get('State', None) == 'running'

    def non_terminated_nodes(self, tag_filters):
        self._update_nodes()
        return super(FakeMultiNodeDockerProvider, self).non_terminated_nodes(tag_filters)

    def is_running(self, node_id):
        with self.lock:
            self._update_nodes()
            return node_id in self._nodes and self._is_docker_running(node_id)

    def is_terminated(self, node_id):
        with self.lock:
            self._update_nodes()
            return node_id not in self._nodes and (not self._is_docker_running(node_id))

    def get_command_runner(self, log_prefix: str, node_id: str, auth_config: Dict[str, Any], cluster_name: str, process_runner: ModuleType, use_internal_ip: bool, docker_config: Optional[Dict[str, Any]]=None) -> CommandRunnerInterface:
        if self.in_docker_container:
            return super(FakeMultiNodeProvider, self).get_command_runner(log_prefix, node_id, auth_config, cluster_name, process_runner, use_internal_ip)
        common_args = {'log_prefix': log_prefix, 'node_id': node_id, 'provider': self, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': use_internal_ip}
        docker_config['container_name'] = self._container_name(node_id)
        docker_config['image'] = self._docker_image
        return FakeDockerCommandRunner(docker_config, **common_args)

    def _get_ip(self, node_id: str) -> Optional[str]:
        for i in range(3):
            self._update_docker_status()
            ip = self._docker_status.get(node_id, {}).get('IP', None)
            if ip:
                return ip
            time.sleep(3)
        return None

    def set_node_tags(self, node_id, tags):
        assert node_id in self._nodes
        self._nodes[node_id]['tags'].update(tags)

    def create_node_with_resources_and_labels(self, node_config, tags, count, resources, labels):
        with self.lock:
            is_head = tags[TAG_RAY_NODE_KIND] == NODE_KIND_HEAD
            if is_head:
                next_id = FAKE_HEAD_NODE_ID
            else:
                next_id = self._next_hex_node_id()
            self._nodes[next_id] = {'tags': tags, 'node_spec': self._create_node_spec_with_resources(head=is_head, node_id=next_id, resources=resources)}
            self._update_docker_compose_config()
            self._save_node_state()

    def create_node(self, node_config: Dict[str, Any], tags: Dict[str, str], count: int) -> Optional[Dict[str, Any]]:
        resources = self._head_resources
        return self.create_node_with_resources_and_labels(node_config, tags, count, resources, {})

    def _terminate_node(self, node):
        self._update_docker_compose_config()
        self._save_node_state()

    @staticmethod
    def bootstrap_config(cluster_config):
        return cluster_config
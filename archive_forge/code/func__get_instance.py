import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.tags import TAG_RAY_USER_NODE_TYPE
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance
def _get_instance(self, cloud_instance_id: str) -> Instance:
    instance = Instance()
    instance.cloud_instance_id = cloud_instance_id
    if self._provider.is_running(cloud_instance_id):
        instance.status = Instance.ALLOCATED
    elif self._provider.is_terminated(cloud_instance_id):
        instance.status = Instance.STOPPED
    else:
        instance.status = Instance.UNKNOWN
    instance.internal_ip = self._provider.internal_ip(cloud_instance_id)
    instance.external_ip = self._provider.external_ip(cloud_instance_id)
    instance.instance_type = self._provider.node_tags(cloud_instance_id)[TAG_RAY_USER_NODE_TYPE]
    return instance
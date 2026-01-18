import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
def _reconcile_with_node_provider(self) -> None:
    none_terminated_cloud_instances = self._node_provider.get_non_terminated_nodes()
    stopping_instances, _ = self._instance_storage.get_instances(status_filter={Instance.STOPPING})
    for instance in stopping_instances.values():
        if none_terminated_cloud_instances.get(instance.cloud_instance_id) is None:
            instance.status = Instance.STOPPED
            result, _ = self._instance_storage.upsert_instance(instance, expected_instance_version=instance.version)
            if not result:
                logger.warning('Failed to update instance status to STOPPED')
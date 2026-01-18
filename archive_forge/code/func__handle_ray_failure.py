import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
def _handle_ray_failure(self, instance_ids: List[str]) -> int:
    failing_instances, _ = self._instance_storage.get_instances(instance_ids=instance_ids, status_filter={Instance.ALLOCATED}, ray_status_filter={Instance.RAY_STOPPED, Instance.RAY_INSTALL_FAILED})
    if not failing_instances:
        logger.debug('No ray failure')
        return
    failing_instances = failing_instances.values()
    for instance in failing_instances:
        self._node_provider.terminate_node(instance.cloud_instance_id)
        instance.status = Instance.STOPPING
        result, _ = self._instance_storage.upsert_instance(instance, expected_instance_version=instance.version)
        if not result:
            logger.warning('Failed to update instance status to STOPPING')
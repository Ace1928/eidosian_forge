import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler._private.constants import (
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
def _may_launch_new_instances(self):
    new_instances, _ = self._instance_storage.get_instances(status_filter={Instance.UNKNOWN})
    if not new_instances:
        logger.debug('No instances to launch')
        return
    queued_instances = []
    for instance in new_instances.values():
        instance.status = Instance.QUEUED
        success, version = self._instance_storage.upsert_instance(instance, expected_instance_version=instance.version)
        if success:
            instance.version = version
            queued_instances.append(instance)
        else:
            logger.error(f'Failed to update {instance} QUEUED')
    instances_by_type = defaultdict(list)
    for instance in queued_instances:
        instances_by_type[instance.instance_type].append(instance)
    for instance_type, instances in instances_by_type.items():
        for i in range(0, len(instances), self._max_instances_per_request):
            self._launch_instance_executor.submit(self._launch_new_instances_by_type, instance_type, instances[i:min(i + self._max_instances_per_request, len(instances))])
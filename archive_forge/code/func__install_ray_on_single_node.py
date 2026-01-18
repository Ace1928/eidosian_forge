import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.ray_installer import RayInstaller
from ray.core.generated.instance_manager_pb2 import Instance
def _install_ray_on_single_node(self, instance: Instance) -> None:
    assert instance.status == Instance.ALLOCATED
    assert instance.ray_status == Instance.RAY_STATUS_UNKOWN
    instance.ray_status = Instance.RAY_INSTALLING
    success, version = self._instance_storage.upsert_instance(instance, expected_instance_version=instance.version)
    if not success:
        logger.warning(f'Failed to update instance {instance.instance_id} to RAY_INSTALLING')
        return
    installed = False
    backoff_factor = 1
    for _ in range(self._max_install_attempts):
        installed = self._ray_installer.install_ray(instance, self._head_node_ip)
        if installed:
            break
        logger.warning('Failed to install ray, retrying...')
        time.sleep(self._install_retry_interval * backoff_factor)
        backoff_factor *= 2
    if not installed:
        instance.ray_status = Instance.RAY_INSTALL_FAILED
        success, version = self._instance_storage.upsert_instance(instance, expected_instance_version=version)
    else:
        instance.ray_status = Instance.RAY_RUNNING
        success, version = self._instance_storage.upsert_instance(instance, expected_instance_version=version)
    if not success:
        logger.warning(f'Failed to update instance {instance.instance_id} to {instance.status}')
        return
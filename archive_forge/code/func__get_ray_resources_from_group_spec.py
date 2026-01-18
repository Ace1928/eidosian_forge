import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _get_ray_resources_from_group_spec(group_spec: Dict[str, Any], is_head: bool) -> Dict[str, int]:
    """
    Infers Ray resources from rayStartCommands and K8s limits.
    The resources extracted are used in autoscaling calculations.

    TODO: Expose a better interface in the RayCluster CRD for Ray resource annotations.
    For now, we take the rayStartParams as the primary source of truth.
    """
    ray_start_params = group_spec['rayStartParams']
    k8s_resource_limits = group_spec['template']['spec']['containers'][0].get('resources', {}).get('limits', {})
    group_name = _HEAD_GROUP_NAME if is_head else group_spec['groupName']
    num_cpus = _get_num_cpus(ray_start_params, k8s_resource_limits, group_name)
    num_gpus = _get_num_gpus(ray_start_params, k8s_resource_limits, group_name)
    custom_resource_dict = _get_custom_resources(ray_start_params, group_name)
    memory = _get_memory(ray_start_params, k8s_resource_limits)
    resources = {}
    assert isinstance(num_cpus, int)
    resources['CPU'] = num_cpus
    if num_gpus is not None:
        resources['GPU'] = num_gpus
    if memory is not None:
        resources['memory'] = memory
    resources.update(custom_resource_dict)
    return resources
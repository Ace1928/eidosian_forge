import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _get_num_cpus(ray_start_params: Dict[str, str], k8s_resource_limits: Dict[str, str], group_name: str) -> int:
    """Get CPU annotation from ray_start_params or k8s_resource_limits,
    with priority for ray_start_params.
    """
    if 'num-cpus' in ray_start_params:
        return int(ray_start_params['num-cpus'])
    elif 'cpu' in k8s_resource_limits:
        cpu_quantity: str = k8s_resource_limits['cpu']
        return _round_up_k8s_quantity(cpu_quantity)
    else:
        raise ValueError(f'Autoscaler failed to detect `CPU` resources for group {group_name}.\nSet the `--num-cpus` rayStartParam and/or the CPU resource limit for the Ray container.')
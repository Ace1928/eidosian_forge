import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _get_custom_resources(ray_start_params: Dict[str, Any], group_name: str) -> Dict[str, int]:
    """Format custom resources based on the `resources` Ray start param.

    Currently, the value of the `resources` field must
    be formatted as follows:
    '"{"Custom1": 1, "Custom2": 5}"'.

    This method first converts the input to a correctly formatted
    json string and then loads that json string to a dict.
    """
    if 'resources' not in ray_start_params:
        return {}
    resources_string = ray_start_params['resources']
    try:
        resources_json = resources_string[1:-1].replace('\\', '')
        resources = json.loads(resources_json)
        assert isinstance(resources, dict)
        for key, value in resources.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
    except Exception as e:
        logger.error(f'Error reading `resource` rayStartParam for group {group_name}. For the correct format, refer to example configuration at https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/kuberay/ray-cluster.complete.yaml.')
        raise e
    return resources
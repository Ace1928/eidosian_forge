import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _derive_autoscaling_config_from_ray_cr(ray_cr: Dict[str, Any]) -> Dict[str, Any]:
    provider_config = _generate_provider_config(ray_cr['metadata']['namespace'])
    available_node_types = _generate_available_node_types_from_ray_cr_spec(ray_cr['spec'])
    global_max_workers = sum((node_type['max_workers'] for node_type in available_node_types.values()))
    legacy_autoscaling_fields = _generate_legacy_autoscaling_config_fields()
    autoscaler_options = ray_cr['spec'].get(AUTOSCALER_OPTIONS_KEY, {})
    if IDLE_SECONDS_KEY in autoscaler_options:
        idle_timeout_minutes = autoscaler_options[IDLE_SECONDS_KEY] / 60.0
    else:
        idle_timeout_minutes = 1.0
    if autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_CONSERVATIVE:
        upscaling_speed = 1
    elif autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_DEFAULT:
        upscaling_speed = 1000
    elif autoscaler_options.get(UPSCALING_KEY) == UPSCALING_VALUE_AGGRESSIVE:
        upscaling_speed = 1000
    else:
        upscaling_speed = 1000
    autoscaling_config = {'provider': provider_config, 'cluster_name': ray_cr['metadata']['name'], 'head_node_type': _HEAD_GROUP_NAME, 'available_node_types': available_node_types, 'max_workers': global_max_workers, 'idle_timeout_minutes': idle_timeout_minutes, 'upscaling_speed': upscaling_speed, **legacy_autoscaling_fields}
    validate_config(autoscaling_config)
    return autoscaling_config
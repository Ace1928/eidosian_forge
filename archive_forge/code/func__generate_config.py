import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def _generate_config(self, head_resources, worker_node_types, extra_provider_config, upscaling_speed, idle_timeout_minutes):
    base_config = yaml.safe_load(open(os.path.join(os.path.dirname(ray.__file__), 'autoscaler/spark/defaults.yaml')))
    custom_config = copy.deepcopy(base_config)
    custom_config['available_node_types'] = worker_node_types
    custom_config['available_node_types']['ray.head.default'] = {'resources': head_resources, 'node_config': {}, 'max_workers': 0}
    custom_config['max_workers'] = sum((v['max_workers'] for _, v in worker_node_types.items()))
    custom_config['provider'].update(extra_provider_config)
    custom_config['upscaling_speed'] = upscaling_speed
    custom_config['idle_timeout_minutes'] = idle_timeout_minutes
    return custom_config
import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def _load_config_file(self, config_type: str) -> Dict[str, Any]:
    """load JSON config file"""
    cloudwatch_config = self.provider_config['cloudwatch']
    json_config_file_section = cloudwatch_config.get(config_type, {})
    json_config_file_path = json_config_file_section.get('config', {})
    json_config_path = os.path.abspath(json_config_file_path)
    with open(json_config_path) as f:
        data = json.load(f)
    return data
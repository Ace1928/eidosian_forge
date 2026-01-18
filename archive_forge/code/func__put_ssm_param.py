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
def _put_ssm_param(self, parameter: Dict[str, Any], parameter_name: str) -> None:
    """upload cloudwatch config to the SSM parameter store"""
    self.ssm_client.put_parameter(Name=parameter_name, Type='String', Value=json.dumps(parameter), Overwrite=True, Tier='Intelligent-Tiering')
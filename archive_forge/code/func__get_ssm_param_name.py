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
def _get_ssm_param_name(self, config_type: str) -> str:
    """return the parameter name for cloudwatch configs"""
    ssm_config_param_name = 'AmazonCloudWatch-' + 'ray_{}_config_{}'.format(config_type, self.cluster_name)
    return ssm_config_param_name
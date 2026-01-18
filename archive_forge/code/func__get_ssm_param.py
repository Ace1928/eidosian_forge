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
def _get_ssm_param(self, parameter_name: str) -> str:
    """
        get the SSM parameter value associated with the given parameter name
        """
    response = self.ssm_client.get_parameter(Name=parameter_name)
    logger.info('Successfully fetch ssm parameter: {}'.format(parameter_name))
    res = response.get('Parameter', {})
    cwa_parameter = res.get('Value', {})
    return cwa_parameter
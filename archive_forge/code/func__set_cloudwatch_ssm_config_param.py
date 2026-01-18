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
def _set_cloudwatch_ssm_config_param(self, parameter_name: str, config_type: str) -> str:
    """
        get cloudwatch config for the given param and config type from SSM
        if it exists, put it in the SSM param store if not
        """
    try:
        parameter_value = self._get_ssm_param(parameter_name)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ParameterNotFound':
            logger.info('Cloudwatch {} config file is not found at SSM parameter store. Checking for Unified CloudWatch Agent installation'.format(config_type))
            return self._get_default_empty_config_file_hash()
        else:
            logger.info('Failed to fetch Unified CloudWatch Agent config from SSM parameter store.')
            logger.error(e)
            raise e
    return parameter_value
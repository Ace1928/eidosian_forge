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
def _start_cloudwatch_agent(self, cwa_param_name: str) -> None:
    """start Unified CloudWatch Agent"""
    logger.info('Starting Unified CloudWatch Agent package on node {}.'.format(self.node_id))
    parameters_start_cwa = {'action': ['configure'], 'mode': ['ec2'], 'optionalConfigurationSource': ['ssm'], 'optionalConfigurationLocation': [cwa_param_name], 'optionalRestart': ['yes']}
    self._ssm_command_waiter('AmazonCloudWatch-ManageAgent', parameters_start_cwa, self.node_id)
    logger.info('Unified CloudWatch Agent started successfully on node {}.'.format(self.node_id))
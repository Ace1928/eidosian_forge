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
def _restart_cloudwatch_agent(self) -> None:
    """restart Unified CloudWatch Agent"""
    cwa_param_name = self._get_ssm_param_name(CloudwatchConfigType.AGENT.value)
    logger.info('Restarting Unified CloudWatch Agent package on node {}.'.format(self.node_id))
    self._stop_cloudwatch_agent()
    self._start_cloudwatch_agent(cwa_param_name)
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
def _send_command_to_node(self, document_name: str, parameters: Dict[str, List[str]], node_id: str) -> Dict[str, Any]:
    """send SSM command to the given nodes"""
    logger.debug('Sending SSM command to {} node(s). Document name: {}. Parameters: {}.'.format(node_id, document_name, parameters))
    response = self.ssm_client.send_command(InstanceIds=[node_id], DocumentName=document_name, Parameters=parameters, MaxConcurrency='1', MaxErrors='0')
    return response
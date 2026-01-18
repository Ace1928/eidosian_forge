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
def _update_cloudwatch_hash_tag_value(self, node_id: str, sha1_hash_value: str, config_type: str):
    hash_key_value = '-'.join([CLOUDWATCH_CONFIG_HASH_TAG_BASE, config_type])
    self.ec2_client.create_tags(Resources=[node_id], Tags=[{'Key': hash_key_value, 'Value': sha1_hash_value}])
    logger.info('Successfully update cloudwatch {} hash tag on {}'.format(config_type, node_id))
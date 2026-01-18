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
def _sha1_hash_json(self, value: str) -> str:
    """calculate the json string sha1 hash"""
    sha1_hash = hashlib.new('sha1')
    binary_value = value.encode('ascii')
    sha1_hash.update(binary_value)
    sha1_res = sha1_hash.hexdigest()
    return sha1_res
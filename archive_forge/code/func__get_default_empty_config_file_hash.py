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
def _get_default_empty_config_file_hash(self):
    default_cw_config = '{}'
    parameter_value = self._sha1_hash_json(default_cw_config)
    return parameter_value
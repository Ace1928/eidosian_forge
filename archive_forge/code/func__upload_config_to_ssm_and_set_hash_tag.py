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
def _upload_config_to_ssm_and_set_hash_tag(self, config_type: str):
    data = self.CLOUDWATCH_CONFIG_TYPE_TO_CONFIG_VARIABLE_REPLACE_FUNC.get(config_type)(config_type)
    sha1_hash_value = self._sha1_hash_file(config_type)
    self._upload_config_to_ssm(data, config_type)
    self._update_cloudwatch_hash_tag_value(self.node_id, sha1_hash_value, config_type)
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
def _replace_cwa_config_vars(self, config_type: str) -> Dict[str, Any]:
    """
        replace {instance_id}, {region}, {cluster_name}
        variable occurrences in Unified Cloudwatch Agent config file
        """
    cwa_config = self._load_config_file(config_type)
    self._replace_all_config_variables(cwa_config, self.node_id, self.cluster_name, self.provider_config['region'])
    return cwa_config
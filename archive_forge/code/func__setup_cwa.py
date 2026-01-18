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
def _setup_cwa(self) -> bool:
    cwa_installed = self._check_cwa_installed_ec2_tag()
    if cwa_installed == 'False':
        res_cwa_installed = self._ensure_cwa_installed_ssm(self.node_id)
        return res_cwa_installed
    else:
        return True
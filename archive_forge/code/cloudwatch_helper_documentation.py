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
Check if CloudWatch configuration was specified by the user
        in their cluster config file.

        Specifically, this function checks if a CloudWatch config file is
        specified by the user in their cluster config file.

        Args:
            config: provider section of cluster config file.
            config_type: type of CloudWatch config file.

        Returns:
            True if config file is specified by user.
            False if config file is not specified.
        
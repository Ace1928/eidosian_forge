import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def _key_assert_msg(node_type: str) -> str:
    if node_type == NODE_TYPE_LEGACY_WORKER:
        return '`KeyName` missing for worker nodes.'
    elif node_type == NODE_TYPE_LEGACY_HEAD:
        return '`KeyName` missing for head node.'
    else:
        return f'`KeyName` missing from the `node_config` of node type `{node_type}`.'
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
def _security_groups_in_network_config(config: Dict[str, Any]) -> List[List[str]]:
    """
    Returns all security group IDs found in the given node config's network
    interfaces.

    Args:
        config (Dict[str, Any]): node config
    Returns:
        security_group_ids (List[List[str]]): List of security group ID lists
        for all network interfaces, or an empty list if no network interfaces
        are defined. An empty list is returned for each missing network
        interface security group list.
    """
    return [ni.get('Groups', []) for ni in config.get('NetworkInterfaces', [])]
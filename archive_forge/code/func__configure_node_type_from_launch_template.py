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
def _configure_node_type_from_launch_template(config: Dict[str, Any], node_type: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the given node type's
    node config into the parent node config. Any parameters specified in
    node config override the same parameters in the launch template.

    Args:
        config (Dict[str, Any]): config to bootstrap
        node_type (Dict[str, Any]): node type config to bootstrap
    Returns:
        node_type (Dict[str, Any]): The input config with all launch template
        data merged into the node config of the input node type. If no
        launch template data is found, then the config is returned
        unchanged.
    Raises:
        ValueError: If no launch template is found for the given launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    node_type = copy.deepcopy(node_type)
    node_cfg = node_type['node_config']
    if 'LaunchTemplate' in node_cfg:
        node_type['node_config'] = _configure_node_cfg_from_launch_template(config, node_cfg)
    return node_type
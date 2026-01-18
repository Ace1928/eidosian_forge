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
def _configure_security_group(config):
    security_group_info_src = {}
    _set_config_info(security_group_src=security_group_info_src)
    for node_type_key in config['available_node_types']:
        security_group_info_src[node_type_key] = 'config'
    node_types_to_configure = [node_type_key for node_type_key, node_type in config['available_node_types'].items() if 'SecurityGroupIds' not in node_type['node_config']]
    if not node_types_to_configure:
        return config
    head_node_type = config['head_node_type']
    if config['head_node_type'] in node_types_to_configure:
        node_types_to_configure.remove(head_node_type)
        node_types_to_configure.append(head_node_type)
    security_groups = _upsert_security_groups(config, node_types_to_configure)
    for node_type_key in node_types_to_configure:
        node_config = config['available_node_types'][node_type_key]['node_config']
        sg = security_groups[node_type_key]
        node_config['SecurityGroupIds'] = [sg.id]
        security_group_info_src[node_type_key] = 'default'
    return config
import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def _should_create_new_head(head_node_id: Optional[str], new_launch_hash: str, new_head_node_type: str, provider: NodeProvider) -> bool:
    """Decides whether a new head node needs to be created.

    We need a new head if at least one of the following holds:
    (a) There isn't an existing head node
    (b) The user-submitted head node_config differs from the existing head
        node's node_config.
    (c) The user-submitted head node_type key differs from the existing head
        node's node_type.

    Args:
        head_node_id (Optional[str]): head node id if a head exists, else None
        new_launch_hash: hash of current user-submitted head config
        new_head_node_type: current user-submitted head node-type key

    Returns:
        bool: True if a new Ray head node should be launched, False otherwise
    """
    if not head_node_id:
        return True
    head_tags = provider.node_tags(head_node_id)
    current_launch_hash = head_tags.get(TAG_RAY_LAUNCH_CONFIG)
    current_head_type = head_tags.get(TAG_RAY_USER_NODE_TYPE)
    hashes_mismatch = new_launch_hash != current_launch_hash
    types_mismatch = new_head_node_type != current_head_type
    new_head_required = hashes_mismatch or types_mismatch
    if new_head_required:
        with cli_logger.group('Currently running head node is out-of-date with cluster configuration'):
            if hashes_mismatch:
                cli_logger.print('Current hash is {}, expected {}', cf.bold(current_launch_hash), cf.bold(new_launch_hash))
            if types_mismatch:
                cli_logger.print('Current head node type is {}, expected {}', cf.bold(current_head_type), cf.bold(new_head_node_type))
    return new_head_required
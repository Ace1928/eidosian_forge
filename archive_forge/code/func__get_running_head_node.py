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
def _get_running_head_node(config: Dict[str, Any], printable_config_file: str, override_cluster_name: Optional[str], create_if_needed: bool=False, _provider: Optional[NodeProvider]=None, _allow_uninitialized_state: bool=False) -> str:
    """Get a valid, running head node.
    Args:
        config (Dict[str, Any]): Cluster Config dictionary
        printable_config_file: Used for printing formatted CLI commands.
        override_cluster_name: Passed to `get_or_create_head_node` to
            override the cluster name present in `config`.
        create_if_needed: Create a head node if one is not present.
        _provider: [For testing], a Node Provider to use.
        _allow_uninitialized_state: Whether to return a head node that
            is not 'UP TO DATE'. This is used to allow `ray attach` and
            `ray exec` to debug a cluster in a bad state.

    """
    provider = _provider or _get_node_provider(config['provider'], config['cluster_name'])
    head_node_tags = {TAG_RAY_NODE_KIND: NODE_KIND_HEAD}
    nodes = provider.non_terminated_nodes(head_node_tags)
    head_node = None
    _backup_head_node = None
    for node in nodes:
        node_state = provider.node_tags(node).get(TAG_RAY_NODE_STATUS)
        if node_state == STATUS_UP_TO_DATE:
            head_node = node
        else:
            _backup_head_node = node
            cli_logger.warning(f'Head node ({node}) is in state {node_state}.')
    if head_node is not None:
        return head_node
    elif create_if_needed:
        get_or_create_head_node(config, printable_config_file=printable_config_file, restart_only=False, no_restart=False, yes=True, override_cluster_name=override_cluster_name)
        return _get_running_head_node(config, printable_config_file, override_cluster_name, create_if_needed=False, _allow_uninitialized_state=False)
    else:
        if _allow_uninitialized_state and _backup_head_node is not None:
            cli_logger.warning(f'The head node being returned: {_backup_head_node} is not `up-to-date`. If you are not debugging a startup issue it is recommended to restart this head node with: {{}}', cf.bold(f'  ray down  {printable_config_file}'))
            return _backup_head_node
        raise RuntimeError('Head node of cluster ({}) not found!'.format(config['cluster_name']))
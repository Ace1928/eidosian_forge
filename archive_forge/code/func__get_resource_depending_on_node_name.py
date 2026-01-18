import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
def _get_resource_depending_on_node_name(self, node_name: str) -> GCPResource:
    """Return the resource responsible for the node, based on node_name.

        This expects the name to be in format '[NAME]-[UUID]-[TYPE]',
        where [TYPE] is either 'compute' or 'tpu' (see ``GCPNodeType``).
        """
    return self.resources[GCPNodeType.name_to_type(node_name)]
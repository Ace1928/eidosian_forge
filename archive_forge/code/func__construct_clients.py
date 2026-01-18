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
def _construct_clients(self):
    _, _, compute, tpu = construct_clients_from_provider_config(self.provider_config)
    self.resources: Dict[GCPNodeType, GCPResource] = {}
    self.resources[GCPNodeType.COMPUTE] = GCPCompute(compute, self.provider_config['project_id'], self.provider_config['availability_zone'], self.cluster_name)
    if tpu is not None:
        self.resources[GCPNodeType.TPU] = GCPTPU(tpu, self.provider_config['project_id'], self.provider_config['availability_zone'], self.cluster_name)
import asyncio
import logging
import marshal
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import run_background_task
from ray._raylet import GcsClient
from ray.actor import ActorHandle
from ray.serve._private.application_state import ApplicationStateManager
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.proxy_state import ProxyStateManager
from ray.serve._private.storage.kv_store import RayInternalKVStore
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet
from ray.serve.schema import (
from ray.util import metrics
def _read_config_checkpoint(self) -> Tuple[float, Optional[ServeDeploySchema], Optional[TargetCapacityDirection]]:
    """Reads the current Serve config checkpoint.

        The Serve config checkpoint stores active application configs and
        other metadata.

        Returns:

        If the GCS contains a checkpoint, tuple of:
            1. A deployment timestamp.
            2. A Serve config. This Serve config is reconstructed from the
                active application states. It may not exactly match the
                submitted config (e.g. the top-level http options may be
                different).
            3. The target_capacity direction calculated after the Serve
               was submitted.

        If the GCS doesn't contain a checkpoint, returns (0, None, None).
        """
    checkpoint = self.kv_store.get(CONFIG_CHECKPOINT_KEY)
    if checkpoint is not None:
        deployment_time, target_capacity, target_capacity_direction, config_checkpoints_dict = pickle.loads(checkpoint)
        return (deployment_time, ServeDeploySchema(applications=list(config_checkpoints_dict.values()), target_capacity=target_capacity), target_capacity_direction)
    else:
        return (0.0, None, None)
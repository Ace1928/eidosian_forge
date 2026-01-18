import json
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import ObjectRef, cloudpickle
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError, RuntimeEnvSetupError
from ray.serve import metrics
from ray.serve._private import default_impl
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion, VersionedReplica
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import (
from ray.util.placement_group import PlacementGroup
def _check_and_stop_outdated_version_replicas(self) -> bool:
    """Stops replicas with outdated versions to implement rolling updates.

        This includes both explicit code version updates and changes to the
        user_config.

        Returns whether any replicas were stopped.
        """
    if self._target_state.target_num_replicas == 0:
        return False
    old_running_replicas = self._replicas.count(exclude_version=self._target_state.version, states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RUNNING])
    old_stopping_replicas = self._replicas.count(exclude_version=self._target_state.version, states=[ReplicaState.STOPPING])
    new_running_replicas = self._replicas.count(version=self._target_state.version, states=[ReplicaState.RUNNING])
    if self._target_state.target_num_replicas < old_running_replicas + old_stopping_replicas:
        return False
    pending_replicas = self._target_state.target_num_replicas - new_running_replicas - old_running_replicas
    rollout_size = max(int(0.2 * self._target_state.target_num_replicas), 1)
    max_to_stop = max(rollout_size - pending_replicas, 0)
    return self._stop_or_update_outdated_version_replicas(max_to_stop)
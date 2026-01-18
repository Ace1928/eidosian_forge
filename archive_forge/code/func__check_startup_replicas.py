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
def _check_startup_replicas(self, original_state: ReplicaState, stop_on_slow=False) -> List[Tuple[DeploymentReplica, ReplicaStartupStatus]]:
    """
        Common helper function for startup actions tracking and status
        transition: STARTING, UPDATING and RECOVERING.

        Args:
            stop_on_slow: If we consider a replica failed upon observing it's
                slow to reach running state.
        """
    slow_replicas = []
    replicas_failed = False
    for replica in self._replicas.pop(states=[original_state]):
        start_status, error_msg = replica.check_started()
        if start_status == ReplicaStartupStatus.SUCCEEDED:
            self._replicas.add(ReplicaState.RUNNING, replica)
            self._deployment_scheduler.on_replica_running(self._id, replica.replica_tag, replica.actor_node_id)
            logger.info(f'Replica {replica.replica_tag} started successfully on node {replica.actor_node_id}.', extra={'log_to_stderr': False})
        elif start_status == ReplicaStartupStatus.FAILED:
            if self._replica_constructor_retry_counter >= 0:
                self._replica_constructor_retry_counter += 1
                self._replica_constructor_error_msg = error_msg
            replicas_failed = True
            self._stop_replica(replica)
        elif start_status in [ReplicaStartupStatus.PENDING_ALLOCATION, ReplicaStartupStatus.PENDING_INITIALIZATION]:
            is_slow = time.time() - replica._start_time > SLOW_STARTUP_WARNING_S
            if is_slow:
                slow_replicas.append((replica, start_status))
            if is_slow and stop_on_slow:
                self._stop_replica(replica, graceful_stop=False)
            else:
                self._replicas.add(original_state, replica)
    failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
    if replicas_failed and self._replica_constructor_retry_counter > failed_to_start_threshold:
        self._backoff_time_s = min(EXPONENTIAL_BACKOFF_FACTOR * self._backoff_time_s, MAX_BACKOFF_TIME_S)
    return slow_replicas
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
def _check_curr_status(self) -> Tuple[bool, bool]:
    """Check the current deployment status.

        Checks the difference between the target vs. running replica count for
        the target version.

        This will update the current deployment status depending on the state
        of the replicas.

        Returns (deleted, any_replicas_recovering).
        """
    target_version = self._target_state.version
    any_replicas_recovering = self._replicas.count(states=[ReplicaState.RECOVERING]) > 0
    all_running_replica_cnt = self._replicas.count(states=[ReplicaState.RUNNING])
    running_at_target_version_replica_cnt = self._replicas.count(states=[ReplicaState.RUNNING], version=target_version)
    failed_to_start_count = self._replica_constructor_retry_counter
    failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
    if failed_to_start_count >= failed_to_start_threshold and failed_to_start_threshold != 0:
        if running_at_target_version_replica_cnt > 0:
            self._replica_constructor_retry_counter = -1
        else:
            self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UNHEALTHY, status_trigger=DeploymentStatusTrigger.REPLICA_STARTUP_FAILED, message=f'The deployment failed to start {failed_to_start_count} times in a row. This may be due to a problem with its constructor or initial health check failing. See controller logs for details. Retrying after {self._backoff_time_s} seconds. Error:\n{self._replica_constructor_error_msg}')
            return (False, any_replicas_recovering)
    if self._replicas.count(states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RECOVERING, ReplicaState.STOPPING]) == 0:
        if self._target_state.deleting and all_running_replica_cnt == 0:
            return (True, any_replicas_recovering)
        if self._target_state.target_num_replicas == running_at_target_version_replica_cnt and running_at_target_version_replica_cnt == all_running_replica_cnt:
            if self._curr_status_info.status == DeploymentStatus.UPSCALING:
                status_trigger = DeploymentStatusTrigger.UPSCALE_COMPLETED
            elif self._curr_status_info.status == DeploymentStatus.DOWNSCALING:
                status_trigger = DeploymentStatusTrigger.DOWNSCALE_COMPLETED
            elif self._curr_status_info.status == DeploymentStatus.UPDATING and self._curr_status_info.status_trigger == DeploymentStatusTrigger.CONFIG_UPDATE_STARTED:
                status_trigger = DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED
            elif self._curr_status_info.status == DeploymentStatus.UNHEALTHY:
                status_trigger = DeploymentStatusTrigger.UNSPECIFIED
            else:
                status_trigger = self._curr_status_info.status_trigger
            self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.HEALTHY, status_trigger=status_trigger)
            return (False, any_replicas_recovering)
    return (False, any_replicas_recovering)
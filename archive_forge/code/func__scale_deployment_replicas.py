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
def _scale_deployment_replicas(self) -> Tuple[List[ReplicaSchedulingRequest], DeploymentDownscaleRequest]:
    """Scale the given deployment to the number of replicas."""
    assert self._target_state.target_num_replicas >= 0, 'Target number of replicas must be greater than or equal to 0.'
    upscale = []
    downscale = None
    self._check_and_stop_outdated_version_replicas()
    current_replicas = self._replicas.count(states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RUNNING])
    recovering_replicas = self._replicas.count(states=[ReplicaState.RECOVERING])
    delta_replicas = self._target_state.target_num_replicas - current_replicas - recovering_replicas
    if delta_replicas == 0:
        return (upscale, downscale)
    elif delta_replicas > 0:
        stopping_replicas = self._replicas.count(states=[ReplicaState.STOPPING])
        to_add = max(delta_replicas - stopping_replicas, 0)
        if to_add > 0:
            failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
            if self._replica_constructor_retry_counter >= failed_to_start_threshold:
                if time.time() - self._last_retry < self._backoff_time_s + random.uniform(0, 3):
                    return (upscale, downscale)
            self._last_retry = time.time()
            logger.info(f"Adding {to_add} replica{('s' if to_add > 1 else '')} to deployment {self.deployment_name} in application '{self.app_name}'.")
            for _ in range(to_add):
                replica_name = ReplicaName(self.app_name, self.deployment_name, get_random_letters())
                new_deployment_replica = DeploymentReplica(self._controller_name, replica_name.replica_tag, self._id, self._target_state.version)
                upscale.append(new_deployment_replica.start(self._target_state.info))
                self._replicas.add(ReplicaState.STARTING, new_deployment_replica)
                logger.debug(f"Adding STARTING to replica_tag: {replica_name}, deployment: '{self.deployment_name}', application: '{self.app_name}'")
    elif delta_replicas < 0:
        to_remove = -delta_replicas
        logger.info(f"Removing {to_remove} replica{('s' if to_remove > 1 else '')} from deployment '{self.deployment_name}' in application '{self.app_name}'.")
        downscale = DeploymentDownscaleRequest(deployment_id=self._id, num_to_stop=to_remove)
    return (upscale, downscale)
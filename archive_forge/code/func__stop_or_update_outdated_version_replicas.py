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
def _stop_or_update_outdated_version_replicas(self, max_to_stop=math.inf) -> bool:
    """Stop or update replicas with outdated versions.

        Stop replicas with versions that require the actor to be restarted, and
        reconfigure replicas that require refreshing deployment config values.

        Args:
            max_to_stop: max number of replicas to stop, by default,
            it will stop all replicas with outdated version.
        """
    replicas_to_update = self._replicas.pop(exclude_version=self._target_state.version, states=[ReplicaState.STARTING, ReplicaState.RUNNING])
    replicas_changed = False
    code_version_changes = 0
    reconfigure_changes = 0
    for replica in replicas_to_update:
        if code_version_changes + reconfigure_changes >= max_to_stop:
            self._replicas.add(replica.actor_details.state, replica)
        elif replica.version.requires_actor_restart(self._target_state.version):
            code_version_changes += 1
            graceful_stop = replica.actor_details.state == ReplicaState.RUNNING
            self._stop_replica(replica, graceful_stop=graceful_stop)
            replicas_changed = True
        elif replica.actor_details.state == ReplicaState.RUNNING:
            reconfigure_changes += 1
            if replica.version.requires_long_poll_broadcast(self._target_state.version):
                replicas_changed = True
            actor_updating = replica.reconfigure(self._target_state.version)
            if actor_updating:
                self._replicas.add(ReplicaState.UPDATING, replica)
            else:
                self._replicas.add(ReplicaState.RUNNING, replica)
            logger.debug(f'Adding UPDATING to replica_tag: {replica.replica_tag}, deployment_name: {self.deployment_name}, app_name: {self.app_name}')
        else:
            self._replicas.add(replica.actor_details.state, replica)
    if code_version_changes > 0:
        logger.info(f"Stopping {code_version_changes} replicas of deployment '{self.deployment_name}' in application '{self.app_name}' with outdated versions.")
    if reconfigure_changes > 0:
        logger.info(f"Updating {reconfigure_changes} replicas of deployment '{self.deployment_name}' in application '{self.app_name}' with outdated deployment configs.")
        ServeUsageTag.USER_CONFIG_LIGHTWEIGHT_UPDATED.record('True')
    return replicas_changed
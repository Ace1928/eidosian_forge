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
class DeploymentStateManager:
    """Manages all state for deployments in the system.

    This class is *not* thread safe, so any state-modifying methods should be
    called with a lock held.
    """

    def __init__(self, controller_name: str, kv_store: KVStoreBase, long_poll_host: LongPollHost, all_current_actor_names: List[str], all_current_placement_group_names: List[str], cluster_node_info_cache: ClusterNodeInfoCache, head_node_id_override: Optional[str]=None):
        self._controller_name = controller_name
        self._kv_store = kv_store
        self._long_poll_host = long_poll_host
        self._cluster_node_info_cache = cluster_node_info_cache
        self._deployment_scheduler = default_impl.create_deployment_scheduler(cluster_node_info_cache, head_node_id_override)
        self._deployment_states: Dict[DeploymentID, DeploymentState] = dict()
        self._recover_from_checkpoint(all_current_actor_names, all_current_placement_group_names)
        self.handle_metrics_store = InMemoryMetricsStore()

    def _create_deployment_state(self, deployment_id):
        self._deployment_scheduler.on_deployment_created(deployment_id, SpreadDeploymentSchedulingPolicy())
        return DeploymentState(deployment_id, self._controller_name, self._long_poll_host, self._deployment_scheduler, self._cluster_node_info_cache, self._save_checkpoint_func)

    def record_autoscaling_metrics(self, data, send_timestamp: float):
        replica_tag, window_avg = data
        if window_avg is not None:
            replica_name = ReplicaName.from_replica_tag(replica_tag)
            self._deployment_states[replica_name.deployment_id].record_autoscaling_metrics(replica_tag, window_avg)

    def record_handle_metrics(self, data: Dict[str, float], send_timestamp: float):
        self.handle_metrics_store.add_metrics_point(data, send_timestamp)

    def get_autoscaling_metrics(self):
        """
        Return autoscaling metrics (used for dumping from controller)
        """
        return {deployment: deployment_state.replica_average_ongoing_requests for deployment, deployment_state in self._deployment_states.items()}

    def _map_actor_names_to_deployment(self, all_current_actor_names: List[str]) -> Dict[str, List[str]]:
        """
        Given a list of all actor names queried from current ray cluster,
        map them to corresponding deployments.

        Example:
            Args:
                [A#zxc123, B#xcv234, A#qwe234]
            Returns:
                {
                    A: [A#zxc123, A#qwe234]
                    B: [B#xcv234]
                }
        """
        all_replica_names = [actor_name for actor_name in all_current_actor_names if ReplicaName.is_replica_name(actor_name)]
        deployment_to_current_replicas = defaultdict(list)
        if len(all_replica_names) > 0:
            for replica_name in all_replica_names:
                replica_tag = ReplicaName.from_str(replica_name)
                deployment_to_current_replicas[replica_tag.deployment_id].append(replica_name)
        return deployment_to_current_replicas

    def _detect_and_remove_leaked_placement_groups(self, all_current_actor_names: List[str], all_current_placement_group_names: List[str]):
        """Detect and remove any placement groups not associated with a replica.

        This can happen under certain rare circumstances:
            - The controller creates a placement group then crashes before creating
            the associated replica actor.
            - While the controller is down, a replica actor crashes but its placement
            group still exists.

        In both of these (or any other unknown cases), we simply need to remove the
        leaked placement groups.
        """
        leaked_pg_names = []
        for pg_name in all_current_placement_group_names:
            if ReplicaName.is_replica_name(pg_name) and pg_name not in all_current_actor_names:
                leaked_pg_names.append(pg_name)
        if len(leaked_pg_names) > 0:
            logger.warning(f'Detected leaked placement groups: {leaked_pg_names}. The placement groups will be removed. This can happen in rare circumstances when the controller crashes and should not cause any issues. If this happens repeatedly, please file an issue on GitHub.')
        for leaked_pg_name in leaked_pg_names:
            try:
                pg = ray.util.get_placement_group(leaked_pg_name)
                ray.util.remove_placement_group(pg)
            except Exception:
                logger.exception(f'Failed to remove leaked placement group {leaked_pg_name}.')

    def _recover_from_checkpoint(self, all_current_actor_names: List[str], all_current_placement_group_names: List[str]):
        """
        Recover from checkpoint upon controller failure with all actor names
        found in current cluster.

        Each deployment resumes target state from checkpoint if available.

        For current state it will prioritize reconstructing from current
        actor names found that matches deployment tag if applicable.
        """
        self._detect_and_remove_leaked_placement_groups(all_current_actor_names, all_current_placement_group_names)
        deployment_to_current_replicas = self._map_actor_names_to_deployment(all_current_actor_names)
        checkpoint = self._kv_store.get(CHECKPOINT_KEY)
        if checkpoint is not None:
            deployment_state_info = cloudpickle.loads(checkpoint)
            for deployment_id, checkpoint_data in deployment_state_info.items():
                deployment_state = self._create_deployment_state(deployment_id)
                deployment_state.recover_target_state_from_checkpoint(checkpoint_data)
                if len(deployment_to_current_replicas[deployment_id]) > 0:
                    deployment_state.recover_current_state_from_replica_actor_names(deployment_to_current_replicas[deployment_id])
                self._deployment_states[deployment_id] = deployment_state

    def shutdown(self):
        """
        Shutdown all running replicas by notifying the controller, and leave
        it to the controller event loop to take actions afterwards.

        Once shutdown signal is received, it will also prevent any new
        deployments or replicas from being created.

        One can send multiple shutdown signals but won't effectively make any
        difference compare to calling it once.
        """
        for deployment_state in self._deployment_states.values():
            deployment_state.delete()
        self._kv_store.delete(CHECKPOINT_KEY)

    def is_ready_for_shutdown(self) -> bool:
        """Return whether all deployments are shutdown.

        Check there are no deployment states and no checkpoints.
        """
        return len(self._deployment_states) == 0 and self._kv_store.get(CHECKPOINT_KEY) is None

    def _save_checkpoint_func(self, *, writeahead_checkpoints: Optional[Dict[str, Tuple]]) -> None:
        """Write a checkpoint of all deployment states.
        By default, this checkpoints the current in-memory state of each
        deployment. However, these can be overwritten by passing
        `writeahead_checkpoints` in order to checkpoint an update before
        applying it to the in-memory state.
        """
        deployment_state_info = {deployment_id: deployment_state.get_checkpoint_data() for deployment_id, deployment_state in self._deployment_states.items()}
        if writeahead_checkpoints is not None:
            deployment_state_info.update(writeahead_checkpoints)
        self._kv_store.put(CHECKPOINT_KEY, cloudpickle.dumps(deployment_state_info))

    def get_running_replica_infos(self) -> Dict[DeploymentID, List[RunningReplicaInfo]]:
        return {id: deployment_state.get_running_replica_infos() for id, deployment_state in self._deployment_states.items()}

    def get_deployment_infos(self) -> Dict[DeploymentID, DeploymentInfo]:
        infos: Dict[DeploymentID, DeploymentInfo] = {}
        for deployment_id, deployment_state in self._deployment_states.items():
            infos[deployment_id] = deployment_state.target_info
        return infos

    def get_deployment(self, deployment_id: DeploymentID) -> Optional[DeploymentInfo]:
        if deployment_id in self._deployment_states:
            return self._deployment_states[deployment_id].target_info
        else:
            return None

    def get_deployment_details(self, id: DeploymentID) -> Optional[DeploymentDetails]:
        """Gets detailed info on a deployment.

        Returns:
            DeploymentDetails: if the deployment is live.
            None: if the deployment is deleted.
        """
        statuses = self.get_deployment_statuses([id])
        if len(statuses) == 0:
            return None
        else:
            status_info = statuses[0]
            return DeploymentDetails(name=id.name, status=status_info.status, status_trigger=status_info.status_trigger, message=status_info.message, deployment_config=_deployment_info_to_schema(id.name, self.get_deployment(id)), replicas=self._deployment_states[id].list_replica_details())

    def get_deployment_statuses(self, ids: List[DeploymentID]=None) -> List[DeploymentStatusInfo]:
        statuses = []
        for id, state in self._deployment_states.items():
            if not ids or id in ids:
                statuses.append(state.curr_status_info)
        return statuses

    def deploy(self, deployment_id: DeploymentID, deployment_info: DeploymentInfo) -> bool:
        """Deploy the deployment.

        If the deployment already exists with the same version and config,
        this is a no-op and returns False.

        Returns:
            bool: Whether or not the deployment is being updated.
        """
        if deployment_id not in self._deployment_states:
            self._deployment_states[deployment_id] = self._create_deployment_state(deployment_id)
            self._record_deployment_usage()
        return self._deployment_states[deployment_id].deploy(deployment_info)

    def get_deployments_in_application(self, app_name: str) -> List[str]:
        """Return list of deployment names in application."""
        deployments = []
        for deployment_id in self._deployment_states:
            if deployment_id.app == app_name:
                deployments.append(deployment_id.name)
        return deployments

    def delete_deployment(self, id: DeploymentID):
        if id in self._deployment_states:
            self._deployment_states[id].delete()

    def get_handle_queueing_metrics(self, deployment_id: DeploymentID, look_back_period_s) -> int:
        """
        Return handle queue length metrics
        Args:
            deployment_id: deployment identifier
            look_back_period_s: the look back time period to collect the requests
                metrics
        Returns:
            if multiple handles queue length, return the max number of queue length.
        """
        current_handle_queued_queries = self.handle_metrics_store.max(deployment_id, time.time() - look_back_period_s)
        if current_handle_queued_queries is None:
            current_handle_queued_queries = 0
        return current_handle_queued_queries

    def update(self) -> bool:
        """Updates the state of all deployments to match their goal state.

        Returns True if any of the deployments have replicas in the RECOVERING state.
        """
        deleted_ids = []
        any_recovering = False
        upscales = {}
        downscales = {}
        for deployment_id, deployment_state in self._deployment_states.items():
            if deployment_state.should_autoscale():
                current_handle_queued_queries = self.get_handle_queueing_metrics(deployment_id, deployment_state.get_autoscale_metric_lookback_period())
                deployment_state.autoscale(current_handle_queued_queries)
            deployment_state_update_result = deployment_state.update()
            if deployment_state_update_result.upscale:
                upscales[deployment_id] = deployment_state_update_result.upscale
            if deployment_state_update_result.downscale:
                downscales[deployment_id] = deployment_state_update_result.downscale
            if deployment_state_update_result.deleted:
                deleted_ids.append(deployment_id)
            any_recovering |= deployment_state_update_result.any_replicas_recovering
        deployment_to_replicas_to_stop = self._deployment_scheduler.schedule(upscales, downscales)
        for deployment_id, replicas_to_stop in deployment_to_replicas_to_stop.items():
            self._deployment_states[deployment_id].stop_replicas(replicas_to_stop)
        for deployment_state in self._deployment_states.values():
            deployment_state.notify_running_replicas_changed()
        for deployment_id in deleted_ids:
            self._deployment_scheduler.on_deployment_deleted(deployment_id)
            del self._deployment_states[deployment_id]
        if len(deleted_ids):
            self._record_deployment_usage()
        return any_recovering

    def _record_deployment_usage(self):
        ServeUsageTag.NUM_DEPLOYMENTS.record(str(len(self._deployment_states)))
        num_gpu_deployments = 0
        for deployment_state in self._deployment_states.values():
            if deployment_state.target_info is not None and deployment_state.target_info.replica_config is not None and (deployment_state.target_info.replica_config.ray_actor_options is not None) and (deployment_state.target_info.replica_config.ray_actor_options.get('num_gpus', 0) > 0):
                num_gpu_deployments += 1
        ServeUsageTag.NUM_GPU_DEPLOYMENTS.record(str(num_gpu_deployments))

    def record_multiplexed_replica_info(self, info: MultiplexedReplicaInfo):
        """
        Record multiplexed model ids for a multiplexed replica.

        Args:
            info: Multiplexed replica info including deployment name,
                replica tag and model ids.
        """
        if info.deployment_id not in self._deployment_states:
            app_msg = f" in application '{info.deployment_id.app}'"
            logger.error(f'Deployment {info.deployment_id.name}{app_msg} not found in state manager.')
            return
        self._deployment_states[info.deployment_id].record_multiplexed_model_ids(info.replica_tag, info.model_ids)

    def get_active_node_ids(self) -> Set[str]:
        """Return set of node ids with running replicas of any deployment.

        This is used to determine which node has replicas. Only nodes with replicas and
        head node should have active proxies.
        """
        node_ids = set()
        for deployment_state in self._deployment_states.values():
            node_ids.update(deployment_state.get_active_node_ids())
        return node_ids
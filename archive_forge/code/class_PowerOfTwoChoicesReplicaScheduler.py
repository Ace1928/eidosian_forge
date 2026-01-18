import asyncio
import enum
import logging
import math
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
import ray
from ray._private.utils import load_class
from ray.actor import ActorHandle
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.exceptions import RayActorError
from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.utils import JavaActorHandleProxy, MetricsPusher
from ray.serve.generated.serve_pb2 import DeploymentRoute
from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
from ray.serve.grpc_util import RayServegRPCContext
from ray.util import metrics
class PowerOfTwoChoicesReplicaScheduler(ReplicaScheduler):
    """Chooses a replica for each request using the "power of two choices" procedure.

    Requests are scheduled in FIFO order.

    When a request comes in, two candidate replicas are chosen randomly. Each replica
    is sent a control message to fetch its queue length.

    The replica responds with two items: (queue_length, accepted). Only replicas that
    accept the request are considered; between those, the one with the lower queue
    length is chosen.

    In the case when neither replica accepts the request (e.g., their queues are full),
    the procedure is repeated with backoff. This backoff repeats indefinitely until a
    replica is chosen, so the caller should use timeouts and cancellation to avoid
    hangs.

    Each request being scheduled may spawn an independent task that runs the scheduling
    procedure concurrently. This task will not necessarily satisfy the request that
    started it (in order to maintain the FIFO order). The total number of tasks is
    capped at (2 * num_replicas).
    """
    backoff_sequence_s = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
    queue_len_response_deadline_s = RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S
    max_queue_len_response_deadline_s = RAY_SERVE_MAX_QUEUE_LENGTH_RESPONSE_DEADLINE_S
    max_num_scheduling_tasks_cap = 50

    def __init__(self, event_loop: asyncio.AbstractEventLoop, deployment_id: DeploymentID, prefer_local_node_routing: bool=False, prefer_local_az_routing: bool=False, self_node_id: Optional[str]=None, self_actor_id: Optional[str]=None, self_availability_zone: Optional[str]=None):
        self._loop = event_loop
        self._deployment_id = deployment_id
        self._prefer_local_node_routing = prefer_local_node_routing
        self._prefer_local_az_routing = prefer_local_az_routing
        self._self_node_id = self_node_id
        self._self_availability_zone = self_availability_zone
        self._replica_id_set: Set[str] = set()
        self._replicas: Dict[str, ReplicaWrapper] = {}
        self._lazily_constructed_replicas_updated_event: Optional[asyncio.Event] = None
        self._colocated_replica_ids: DefaultDict[LocalityScope, Set[str]] = defaultdict(set)
        self._multiplexed_model_id_to_replica_ids: DefaultDict[Set[str]] = defaultdict(set)
        self._multiplexed_model_id_fallback_match: Set[str] = set()
        self._scheduling_tasks: Set[asyncio.Task] = set()
        self._pending_requests_to_fulfill: Deque[PendingRequest] = deque()
        self._pending_requests_to_schedule: Deque[PendingRequest] = deque()
        self.num_scheduling_tasks_gauge = metrics.Gauge('serve_num_scheduling_tasks', description='The number of request scheduling tasks in the router.', tag_keys=('app', 'deployment', 'actor_id')).set_default_tags({'app': self._deployment_id.app, 'deployment': self._deployment_id.name, 'actor_id': self_actor_id if self_actor_id else ''})
        self.num_scheduling_tasks_gauge.set(0)
        self.num_scheduling_tasks_in_backoff = 0
        self.num_scheduling_tasks_in_backoff_gauge = metrics.Gauge('serve_num_scheduling_tasks_in_backoff', description='The number of request scheduling tasks in the router that are undergoing backoff.', tag_keys=('app', 'deployment', 'actor_id')).set_default_tags({'app': self._deployment_id.app, 'deployment': self._deployment_id.name, 'actor_id': self_actor_id if self_actor_id else ''})
        self.num_scheduling_tasks_in_backoff_gauge.set(self.num_scheduling_tasks_in_backoff)

    @property
    def _replicas_updated_event(self) -> asyncio.Event:
        """Lazily construct `asyncio.Event`.

        See comment for self._lazily_constructed_replicas_updated_event.
        """
        if self._lazily_constructed_replicas_updated_event is None:
            self._lazily_constructed_replicas_updated_event = asyncio.Event()
        return self._lazily_constructed_replicas_updated_event

    @property
    def num_pending_requests(self) -> int:
        """Current number of requests pending assignment."""
        return len(self._pending_requests_to_fulfill)

    @property
    def curr_num_scheduling_tasks(self) -> int:
        """Current number of scheduling tasks running."""
        return len(self._scheduling_tasks)

    @property
    def max_num_scheduling_tasks(self) -> int:
        """Max number of scheduling tasks to run at any time."""
        return min(self.max_num_scheduling_tasks_cap, 2 * len(self._replicas))

    @property
    def target_num_scheduling_tasks(self) -> int:
        """Target number of scheduling tasks to be running based on pending requests.

        This will never exceed `self.max_num_scheduling_tasks`.
        """
        return min(self.num_pending_requests, self.max_num_scheduling_tasks)

    @property
    def curr_replicas(self) -> Dict[str, ReplicaWrapper]:
        return self._replicas

    @property
    def app_name(self) -> str:
        return self._deployment_id.app

    def update_replicas(self, replicas: List[ReplicaWrapper]):
        """Update the set of available replicas to be considered for scheduling.

        When the set of replicas changes, we may spawn additional scheduling tasks
        if there are pending requests.
        """
        new_replicas = {}
        new_replica_id_set = set()
        new_colocated_replica_ids = defaultdict(set)
        new_multiplexed_model_id_to_replica_ids = defaultdict(set)
        for r in replicas:
            new_replicas[r.replica_id] = r
            new_replica_id_set.add(r.replica_id)
            if self._self_node_id is not None and r.node_id == self._self_node_id:
                new_colocated_replica_ids[LocalityScope.NODE].add(r.replica_id)
            if self._self_availability_zone is not None and r.availability_zone == self._self_availability_zone:
                new_colocated_replica_ids[LocalityScope.AVAILABILITY_ZONE].add(r.replica_id)
            for model_id in r.multiplexed_model_ids:
                new_multiplexed_model_id_to_replica_ids[model_id].add(r.replica_id)
        if self._replica_id_set != new_replica_id_set:
            app_msg = f" in application '{self.app_name}'" if self.app_name else ''
            logger.info(f"Got updated replicas for deployment '{self._deployment_id.name}'{app_msg}: {new_replica_id_set}.", extra={'log_to_stderr': False})
        self._replicas = new_replicas
        self._replica_id_set = new_replica_id_set
        self._colocated_replica_ids = new_colocated_replica_ids
        self._multiplexed_model_id_to_replica_ids = new_multiplexed_model_id_to_replica_ids
        self._replicas_updated_event.set()
        self.maybe_start_scheduling_tasks()

    def _get_candidate_multiplexed_replica_ids(self, model_id: str, get_from_all_replicas: bool=False) -> Set[str]:
        """Get multiplexed model candidates from the current replica.

        By default, we will only choose from replicas that have the requested
        multiplexed model id, if not matched, the function will return an empty set.

        If get_from_all_replicas is True, we will choose from all replicas,
        and we will choose all replicas with the least number of multiplexed model
        ids.

        """
        candidates = set()
        if not get_from_all_replicas:
            if model_id in self._multiplexed_model_id_to_replica_ids:
                candidates = self._multiplexed_model_id_to_replica_ids[model_id]
                if len(candidates) > 0:
                    return candidates
            return candidates
        sorted_replicas = sorted(self._replicas.values(), key=lambda x: len(x.multiplexed_model_ids))
        least_num_multiplexed_model_ids = math.inf
        for replica in sorted_replicas:
            if len(replica.multiplexed_model_ids) <= least_num_multiplexed_model_ids:
                candidates.add(replica.replica_id)
                least_num_multiplexed_model_ids = len(replica.multiplexed_model_ids)
            else:
                break
        return candidates

    async def choose_two_replicas_with_backoff(self, request_metadata: Optional[RequestMetadata]=None) -> AsyncGenerator[List[RunningReplicaInfo], None]:
        """Generator that repeatedly chooses (at most) two random available replicas.

        In the first iteration, only replicas colocated on the same node as this router
        will be considered. If those are occupied, the full set of replicas will be
        considered on subsequent iterations.

        For multiplexing, this will first attempt to choose replicas that have the
        requested model ID for a configured timeout. If no replicas with the matching
        model ID are available after that timeout, it will fall back to the regular
        procedure.

        After each iteration, there will be an increasing backoff sleep time (dictated
        by `self.backoff_sequence_s`). The caller should exit the generator to reset the
        backoff sleep time.
        """
        try:
            entered_backoff = False
            backoff_index = 0
            multiplexed_start_matching_time = None
            multiplexed_matching_timeout = random.uniform(RAY_SERVE_MULTIPLEXED_MODEL_ID_MATCHING_TIMEOUT_S, RAY_SERVE_MULTIPLEXED_MODEL_ID_MATCHING_TIMEOUT_S * 2)
            tried_same_node = False
            tried_same_az = False
            while True:
                while len(self._replicas) == 0:
                    app_msg = f" in application '{self.app_name}'" if self.app_name else ''
                    logger.info(f"Tried to assign replica for deployment '{self._deployment_id.name}'{app_msg} but none are available. Waiting for new replicas to be added.", extra={'log_to_stderr': False})
                    self._replicas_updated_event.clear()
                    await self._replicas_updated_event.wait()
                    logger.info(f"Got replicas for deployment '{self._deployment_id.name}'{app_msg}, waking up.", extra={'log_to_stderr': False})
                if multiplexed_start_matching_time is None:
                    multiplexed_start_matching_time = time.time()
                candidate_replica_ids = None
                if request_metadata is not None and request_metadata.multiplexed_model_id:
                    if time.time() - multiplexed_start_matching_time < multiplexed_matching_timeout:
                        candidate_replica_ids = self._get_candidate_multiplexed_replica_ids(request_metadata.multiplexed_model_id)
                        if len(candidate_replica_ids) == 0 and request_metadata.multiplexed_model_id not in self._multiplexed_model_id_fallback_match:
                            candidate_replica_ids = self._get_candidate_multiplexed_replica_ids(request_metadata.multiplexed_model_id, get_from_all_replicas=True)
                            self._multiplexed_model_id_fallback_match.add(request_metadata.multiplexed_model_id)
                        elif len(candidate_replica_ids) > 0:
                            self._multiplexed_model_id_fallback_match.discard(request_metadata.multiplexed_model_id)
                    else:
                        candidate_replica_ids = self._get_candidate_multiplexed_replica_ids(request_metadata.multiplexed_model_id, get_from_all_replicas=True)
                elif self._prefer_local_node_routing and (not tried_same_node) and (len(self._colocated_replica_ids[LocalityScope.NODE]) > 0):
                    candidate_replica_ids = self._colocated_replica_ids[LocalityScope.NODE]
                    tried_same_node = True
                elif self._prefer_local_az_routing and (not tried_same_az) and (len(self._colocated_replica_ids[LocalityScope.AVAILABILITY_ZONE]) > 0):
                    candidate_replica_ids = self._colocated_replica_ids[LocalityScope.AVAILABILITY_ZONE]
                    tried_same_az = True
                else:
                    candidate_replica_ids = self._replica_id_set
                if candidate_replica_ids:
                    chosen_ids = random.sample(list(candidate_replica_ids), k=min(2, len(candidate_replica_ids)))
                    yield [self._replicas[chosen_id] for chosen_id in chosen_ids]
                if not entered_backoff:
                    entered_backoff = True
                    self.num_scheduling_tasks_in_backoff += 1
                    self.num_scheduling_tasks_in_backoff_gauge.set(self.num_scheduling_tasks_in_backoff)
                await asyncio.sleep(self.backoff_sequence_s[backoff_index])
                backoff_index = min(backoff_index + 1, len(self.backoff_sequence_s) - 1)
        finally:
            if entered_backoff:
                self.num_scheduling_tasks_in_backoff -= 1
                self.num_scheduling_tasks_in_backoff_gauge.set(self.num_scheduling_tasks_in_backoff)

    async def select_from_candidate_replicas(self, candidates: List[ReplicaWrapper], backoff_index: int) -> Optional[ReplicaWrapper]:
        """Chooses the best replica from the list of candidates.

        If none of the replicas can be scheduled, returns `None`.

        The queue length at each replica is queried directly from it. The time waited
        for these queries is capped by a response deadline; if a replica doesn't
        doesn't respond within the deadline it is not considered. The deadline will be
        increased exponentially in backoff.

        Among replicas that respond within the deadline and accept the request (don't
        have full queues), the one with the lowest queue length is chosen.
        """
        max_queue_len_response_deadline_s = max(self.queue_len_response_deadline_s, self.max_queue_len_response_deadline_s)
        queue_len_response_deadline_s = min(self.queue_len_response_deadline_s * 2 ** backoff_index, max_queue_len_response_deadline_s)
        get_queue_state_tasks = []
        for c in candidates:
            t = self._loop.create_task(c.get_queue_state(deadline_s=queue_len_response_deadline_s))
            t.replica_id = c.replica_id
            get_queue_state_tasks.append(t)
        done, pending = await asyncio.wait(get_queue_state_tasks, timeout=queue_len_response_deadline_s, return_when=asyncio.ALL_COMPLETED)
        for t in pending:
            t.cancel()
            logger.warning(f"Failed to get queue length from replica {t.replica_id} within {queue_len_response_deadline_s}s. If this happens repeatedly it's likely caused by high network latency in the cluster. You can configure the deadline using the `RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S` environment variable.")
        chosen_replica_id = None
        lowest_queue_len = math.inf
        for t in done:
            if t.exception() is not None:
                msg = f"Failed to fetch queue length for replica {t.replica_id}: '{t.exception()}'"
                if isinstance(t.exception(), RayActorError):
                    self._replicas.pop(t.replica_id, None)
                    self._replica_id_set.discard(t.replica_id)
                    for id_set in self._colocated_replica_ids.values():
                        id_set.discard(t.replica_id)
                    msg += ' This replica will no longer be considered for requests.'
                logger.warning(msg)
            else:
                queue_len, accepted = t.result()
                if accepted and queue_len < lowest_queue_len:
                    chosen_replica_id = t.replica_id
                    lowest_queue_len = queue_len
        if chosen_replica_id is None:
            return None
        return self._replicas.get(chosen_replica_id, None)

    def _get_pending_request_matching_metadata(self, replica: ReplicaWrapper, request_metadata: Optional[RequestMetadata]=None) -> Optional[PendingRequest]:
        if request_metadata is None or not request_metadata.multiplexed_model_id:
            return None
        for pr in self._pending_requests_to_fulfill:
            if not pr.future.done() and pr.metadata.multiplexed_model_id == request_metadata.multiplexed_model_id:
                return pr
        return None

    def fulfill_next_pending_request(self, replica: ReplicaWrapper, request_metadata: Optional[RequestMetadata]=None):
        """Assign the replica to the next pending request in FIFO order.

        If a pending request has been cancelled, it will be popped from the queue
        and not assigned.
        """
        matched_pending_request = self._get_pending_request_matching_metadata(replica, request_metadata)
        if matched_pending_request is not None:
            matched_pending_request.future.set_result(replica)
            self._pending_requests_to_fulfill.remove(matched_pending_request)
            return
        while len(self._pending_requests_to_fulfill) > 0:
            pr = self._pending_requests_to_fulfill.popleft()
            if not pr.future.done():
                pr.future.set_result(replica)
                break

    def _get_next_pending_request_metadata_to_schedule(self) -> Optional[RequestMetadata]:
        while len(self._pending_requests_to_schedule) > 0:
            pr = self._pending_requests_to_schedule.popleft()
            if not pr.future.done():
                return pr.metadata
        return None

    async def fulfill_pending_requests(self):
        """Repeatedly tries to fulfill a pending request with an available replica.

        This is expected to be run inside a task in self._scheduling tasks.

        When a replica is found, this method will exit if the number of scheduling tasks
        has exceeded the target number. Else it will loop again to schedule another
        replica.
        """
        try:
            while len(self._scheduling_tasks) <= self.target_num_scheduling_tasks:
                backoff_index = 0
                request_metadata = self._get_next_pending_request_metadata_to_schedule()
                async for candidates in self.choose_two_replicas_with_backoff(request_metadata):
                    replica = await self.select_from_candidate_replicas(candidates, backoff_index)
                    if replica is not None:
                        self.fulfill_next_pending_request(replica, request_metadata)
                        break
                    backoff_index += 1
        except Exception:
            logger.exception('Unexpected error in fulfill_pending_requests.')
        finally:
            self._scheduling_tasks.remove(asyncio.current_task(loop=self._loop))
            self.num_scheduling_tasks_gauge.set(self.curr_num_scheduling_tasks)

    def maybe_start_scheduling_tasks(self):
        """Start scheduling tasks to fulfill pending requests if necessary.

        Starts tasks so that there is at least one task per pending request
        (respecting the max number of scheduling tasks).

        In the common case, this will start a single task when a new request comes
        in for scheduling. However, in cases where the number of available replicas
        is updated or a task exits unexpectedly, we may need to start multiple.
        """
        tasks_to_start = self.target_num_scheduling_tasks - self.curr_num_scheduling_tasks
        for _ in range(tasks_to_start):
            self._scheduling_tasks.add(self._loop.create_task(self.fulfill_pending_requests()))
        if tasks_to_start > 0:
            self.num_scheduling_tasks_gauge.set(self.curr_num_scheduling_tasks)

    async def choose_replica_for_query(self, query: Query) -> ReplicaWrapper:
        """Chooses a replica to send the provided request to.

        Requests are scheduled in FIFO order, so this puts a future on the internal
        queue that will be resolved when a replica is available and it's the front of
        the queue.

        Upon cancellation (by the caller), the future is cancelled and will be passed
        over when a replica becomes available.
        """
        pending_request = PendingRequest(asyncio.Future(), query.metadata)
        try:
            self._pending_requests_to_fulfill.append(pending_request)
            self._pending_requests_to_schedule.append(pending_request)
            self.maybe_start_scheduling_tasks()
            replica = await pending_request.future
        except asyncio.CancelledError as e:
            pending_request.future.cancel()
            raise e from None
        return replica

    async def assign_replica(self, query: Query) -> Union[ray.ObjectRef, 'ray._raylet.ObjectRefGenerator']:
        """Choose a replica for the request and send it.

        This will block indefinitely if no replicas are available to handle the
        request, so it's up to the caller to time out or cancel the request.
        """
        replica = await self.choose_replica_for_query(query)
        return replica.send_query(query)
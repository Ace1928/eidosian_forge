import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
class SimpleElasticAgent(ElasticAgent):
    """An ``ElasticAgent`` that manages one particular type of worker role.

    An ``ElasticAgent`` that manages workers (``WorkerGroup``) for a single ``WorkerSpec``
    such as one particular type of worker role.
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float=300):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._total_execution_time = 0

    def get_worker_group(self, role: str=DEFAULT_ROLE) -> WorkerGroup:
        return self._worker_group

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        """Start ``worker_group.spec.local_world_size`` number of workers.

        This is according to worker spec for the worker group .
        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        """Stop all workers in the given worker group.

        Implementors must deal with workers in all states defined by
        ``WorkerState``. That is, it must gracefully handle stopping
        non-existent workers, unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        """Check on the workers for the ``worker_group``.

        This function also returns the new state of the worker group.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _shutdown(self, death_sig: signal.Signals=signal.SIGTERM) -> None:
        """Clean up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
        """
        raise NotImplementedError()

    @staticmethod
    def _set_master_addr_port(store: Store, master_addr: Optional[str], master_port: Optional[int], local_addr: Optional[str]):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]
        if master_addr is None:
            if local_addr:
                master_addr = local_addr
            else:
                master_addr = _get_fq_hostname()
        store.set('MASTER_ADDR', master_addr.encode(encoding='UTF-8'))
        store.set('MASTER_PORT', str(master_port).encode(encoding='UTF-8'))

    @staticmethod
    def _get_master_addr_port(store: Store) -> Tuple[str, int]:
        master_addr = store.get('MASTER_ADDR').decode(encoding='UTF-8')
        master_port = int(store.get('MASTER_PORT').decode(encoding='UTF-8'))
        return (master_addr, master_port)

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        """Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
        spec = worker_group.spec
        store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
        self._store = store
        workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size
        if group_rank == 0:
            self._set_master_addr_port(store, spec.master_addr, spec.master_port, spec.local_addr)
        master_addr, master_port = self._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts
        log.info('[%(role)s] Rendezvous complete for workers. Result:\n  restart_count=%(restart_count)s\n  master_addr=%(master_addr)s\n  master_port=%(master_port)s\n  group_rank=%(group_rank)s\n  group_world_size=%(group_world_size)s\n  local_ranks=%(local_ranks)s\n  role_ranks=%(role_ranks)s\n  global_ranks=%(global_ranks)s\n  role_world_sizes=%(role_world_sizes)s\n  global_world_sizes=%(global_world_sizes)s\n', {'role': spec.role, 'restart_count': restart_count, 'master_addr': master_addr, 'master_port': master_port, 'group_rank': group_rank, 'group_world_size': group_world_size, 'local_ranks': [worker.local_rank for worker in workers], 'role_ranks': [worker.role_rank for worker in workers], 'global_ranks': [worker.global_rank for worker in workers], 'role_world_sizes': [worker.role_world_size for worker in workers], 'global_world_sizes': [worker.world_size for worker in workers]})

    def _get_ranks(self, role_infos: List[_RoleInstanceInfo], role_idx: int, start_idx: int=0, end_idx: int=-1) -> Tuple[int, List[int]]:
        if end_idx == -1:
            end_idx = len(role_infos)
        prefix_sum = 0
        total_sum = 0
        for idx in range(start_idx, end_idx):
            if role_idx > idx:
                prefix_sum += role_infos[idx].local_world_size
            total_sum += role_infos[idx].local_world_size
        return (total_sum, list(range(prefix_sum, prefix_sum + role_infos[role_idx].local_world_size)))

    @prof
    def _assign_worker_ranks(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List[Worker]:
        """Determine proper ranks for worker processes.

        The rank assignment is done according to the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. Each agent retrieves configuration for all agents
           and performs two level sort using role and rank.
        3. Determine the global rank: the global rank of the workers for the current
           agent is the offset of the infos array up to group_rank of the agent.
           The offset is computed as a sum of local_world_size of all agents that
           have rank less than the group_rank. The workers would have the ranks:
           [offset, offset+local_world_size)
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the offset is done from the first
           agent that has the same role as current one and has the minimum group rank.
        """
        role_infos = self._share_and_gather(store, group_rank, group_world_size, spec)
        my_role_info = role_infos[group_rank]
        worker_world_size, worker_global_ranks = self._get_ranks(role_infos, group_rank)
        role_infos = sorted(role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare))
        role_start_idx, role_end_idx = _RoleInstanceInfo.find_role_boundaries(role_infos, my_role_info.role)
        role_pos = next((idx for idx, role_info in enumerate(role_infos) if _RoleInstanceInfo.compare(role_info, my_role_info) == 0))
        role_world_size, role_ranks = self._get_ranks(role_infos, role_pos, role_start_idx, role_end_idx + 1)
        workers = []
        for ind in range(spec.local_world_size):
            worker = Worker(local_rank=ind, global_rank=worker_global_ranks[ind], role_rank=role_ranks[ind], world_size=worker_world_size, role_world_size=role_world_size)
            workers.append(worker)
        return workers

    def _share_and_gather(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List:
        agent_role_info = _RoleInstanceInfo(spec.role, group_rank, spec.local_world_size)
        key_prefix = 'torchelastic/role_info'
        agent_config_enc = agent_role_info.serialize()
        role_infos_bytes = store_util.synchronize(store, agent_config_enc, group_rank, group_world_size, key_prefix)
        role_infos = [_RoleInstanceInfo.deserialize(role_info_bytes) for role_info_bytes in role_infos_bytes]
        return role_infos

    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        """Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        log.info("[%s] Rendezvous'ing worker group", role)
        self._rendezvous(worker_group)
        log.info('[%s] Starting worker group', role)
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id
        worker_group.state = WorkerState.HEALTHY

    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Restart (stops, rendezvous, starts) all local workers in the group."""
        role = worker_group.spec.role
        log.info('[%s] Stopping worker group', role)
        self._stop_workers(worker_group)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    @prof
    def run(self, role: str=DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except SignalException as e:
            log.warning('Received %s death signal, shutting down workers', e.sigval)
            self._shutdown(e.sigval)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown()
            self._total_execution_time = int(time.monotonic() - start_time)

    def get_event_failed(self) -> Event:
        return self._construct_event(state='FAILED', source=EventSource.AGENT, raw_error=traceback.format_exc())

    def get_event_succeeded(self) -> Event:
        return self._construct_event(state='SUCCEEDED', source=EventSource.AGENT)

    def _record_worker_events(self, result: RunResult) -> None:
        for worker in self._worker_group.workers:
            failure = result.failures.get(worker.global_rank)
            state: str = self._get_worker_state(worker, result)
            raw_error = json.dumps(failure.error_file_data) if failure else None
            record(self._construct_event(state, EventSource.WORKER, worker, raw_error))

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        failure = result.failures.get(worker.global_rank)
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and (not failure):
            return 'TERMINATED'
        elif failure or worker.global_rank in result.return_values:
            return result.state.value
        else:
            raise ValueError(f'Unknown worker: {worker.global_rank}')

    def _construct_event(self, state: str, source: EventSource, worker: Optional[Worker]=None, raw_error: Optional[str]=None) -> Event:
        wg = self._worker_group
        spec = wg.spec
        md = {'group_world_size': wg.group_world_size, 'entry_point': spec.get_entrypoint_name()}
        if worker:
            md['local_rank'] = (worker.local_rank,)
            md['role_rank'] = (worker.role_rank,)
            md['role_world_size'] = (worker.role_world_size,)
            global_rank = worker.global_rank
            worker_id = str(worker.id)
        else:
            global_rank = None
            worker_id = None
        md_str = json.dumps(md)
        metadata = {'run_id': spec.rdzv_handler.get_run_id(), 'global_rank': global_rank, 'group_rank': wg.group_rank, 'worker_id': worker_id, 'role': spec.role, 'hostname': _get_fq_hostname(), 'state': state, 'total_run_time': self._total_execution_time, 'rdzv_backend': spec.rdzv_handler.get_backend(), 'raw_error': raw_error, 'metadata': md_str, 'agent_restarts': spec.max_restarts - self._remaining_restarts}
        return Event(f'torchelastic.worker.status.{state}', source=source, metadata=metadata)

    def _record_metrics(self, group_results: RunResult):
        is_failed = group_results.is_failed()
        self._record_flakiness_metric(is_failed)
        spec = self._worker_group.spec
        restarts_happened = self._remaining_restarts != spec.max_restarts
        put_metric(f'workers.{spec.role}.run_total', 1)
        self._record_metric_with_condition('run_success_with_retries', not is_failed and restarts_happened)
        self._record_metric_with_condition('run_success_no_retries', not is_failed and (not restarts_happened))
        self._record_metric_with_condition('run_failed_with_retries', is_failed and restarts_happened)
        self._record_metric_with_condition('run_failed_no_retries', is_failed and (not restarts_happened))

    def _record_metric_with_condition(self, metric_name, condition):
        spec = self._worker_group.spec
        if condition:
            put_metric(f'workers.{spec.role}.{metric_name}', 1)
        else:
            put_metric(f'workers.{spec.role}.{metric_name}', 0)

    def _record_flakiness_metric(self, is_failed: bool=False):
        if is_failed:
            flakiness = 100.0
        else:
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (spec.max_restarts + 1)
        spec = self._worker_group.spec
        put_metric(f'workers.{spec.role}.flakiness', int(flakiness))

    def _invoke_run(self, role: str=DEFAULT_ROLE) -> RunResult:
        spec = self._worker_group.spec
        role = spec.role
        log.info('[%s] starting workers for entrypoint: %s', role, spec.get_entrypoint_name())
        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler
        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state
            put_metric(f'workers.{role}.remaining_restarts', self._remaining_restarts)
            put_metric(f'workers.{role}.{state.name.lower()}', 1)
            if state == WorkerState.SUCCEEDED:
                log.info('[%s] worker group successfully finished. Waiting %s seconds for other agents to finish.', role, self._exit_barrier_timeout)
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    log.info('[%s] Worker group %s. %s/%s attempts left; will restart worker group', role, state.name, self._remaining_restarts, spec.max_restarts)
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    log.info('[%s] Detected %s new nodes from group_rank=%s; will restart worker group', role, num_nodes_waiting, group_rank)
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f'[{role}] Worker group in {state.name} state')

    def _exit_barrier(self):
        """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
        log.info('Local worker group finished (%s). Waiting %s seconds for other agents to finish', self._worker_group.state, self._exit_barrier_timeout)
        start = time.time()
        try:
            store_util.barrier(self._store, self._worker_group.group_rank, self._worker_group.group_world_size, key_prefix=_TERMINAL_STATE_SYNC_ID, barrier_timeout=self._exit_barrier_timeout)
            log.info('Done waiting for other agents. Elapsed: %s seconds', time.time() - start)
        except SignalException as e:
            log.warning('Got termination signal: %s', e.sigval)
            raise
        except Exception:
            log.exception('Error waiting on exit barrier. Elapsed: %s seconds', time.time() - start)
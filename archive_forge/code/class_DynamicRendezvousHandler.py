import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
from .api import (
from .utils import _delay, _PeriodicTimer
class DynamicRendezvousHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes."""
    _node_desc_generator = _NodeDescGenerator()
    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: Optional[_PeriodicTimer]

    @classmethod
    def from_backend(cls, run_id: str, store: Store, backend: RendezvousBackend, min_nodes: int, max_nodes: int, local_addr: Optional[str]=None, timeout: Optional[RendezvousTimeout]=None):
        """Create a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
        """
        node = cls._node_desc_generator.generate(local_addr)
        settings = RendezvousSettings(run_id, min_nodes, max_nodes, timeout or RendezvousTimeout(), keep_alive_interval=timedelta(seconds=5), keep_alive_max_attempt=3)
        state_holder = _BackendRendezvousStateHolder(backend, settings)
        return cls(node, settings, backend.name, store, state_holder)

    def __init__(self, node: _NodeDesc, settings: RendezvousSettings, backend_name: str, store: Store, state_holder: _RendezvousStateHolder) -> None:
        if not settings.run_id:
            raise ValueError('The run id must be a non-empty string.')
        if settings.min_nodes < 1:
            raise ValueError(f'The minimum number of nodes ({settings.min_nodes}) must be greater than zero.')
        if settings.max_nodes < settings.min_nodes:
            raise ValueError(f'The maximum number of nodes ({settings.max_nodes}) must be greater than or equal to the minimum number of nodes ({settings.min_nodes}).')
        self._this_node = node
        self._settings = settings
        self._backend_name = backend_name
        self._store = store
        self._state_holder = state_holder
        self._op_executor = _DistributedRendezvousOpExecutor(self._this_node, self._state_holder, self._settings)
        self._heartbeat_lock = threading.Lock()
        self._keep_alive_timer = None

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING, rank: Optional[int]=None) -> None:
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state, hostname=self._this_node.addr, pid=self._this_node.pid, local_id=self._this_node.local_id, rank=rank)

    @property
    def settings(self) -> RendezvousSettings:
        """Get the settings of the rendezvous."""
        return self._settings

    def get_backend(self) -> str:
        """See base class."""
        return self._backend_name

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        """See base class."""
        msg = f"The node '{self._this_node}' attempts to join the next round of the rendezvous '{self._settings.run_id}'."
        self._record(message=msg)
        log.info(msg)
        try:
            self._stop_heartbeats()
            if self._state_holder.state.round == 0:
                _delay(seconds=(0, 0.3))
            exit_op = _RendezvousExitOp()
            join_op = _RendezvousJoinOp()
            deadline = self._get_deadline(self._settings.timeout.join)
            self._op_executor.run(exit_op, deadline)
            self._op_executor.run(join_op, deadline)
            self._start_heartbeats()
            rank, world_size = self._get_world()
            store = self._get_store()
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise
        msg = f"The node '{self._this_node}' has joined round {self._state_holder.state.round} of the rendezvous '{self._settings.run_id}' as rank {rank} in a world of size {world_size}."
        self._record(message=msg, rank=rank)
        log.info(msg)
        return (store, rank, world_size)

    def is_closed(self) -> bool:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()
                return self._state_holder.state.closed
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def set_closed(self) -> None:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def num_nodes_waiting(self) -> int:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()
                return len(self._state_holder.state.wait_list)
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def get_run_id(self) -> str:
        """See base class."""
        return self._settings.run_id

    def shutdown(self) -> bool:
        """See base class."""
        self._stop_heartbeats()
        try:
            self._close()
            return True
        except RendezvousError as ex:
            msg = f"The node '{self._this_node}' has failed to shutdown the rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
            return False
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def _close(self) -> None:
        op = _RendezvousCloseOp()
        deadline = self._get_deadline(self._settings.timeout.close)
        self._op_executor.run(op, deadline)
        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        self._heartbeat_lock.acquire()
        op = _RendezvousKeepAliveOp()
        deadline = self._get_deadline(self._settings.timeout.heartbeat)
        try:
            self._op_executor.run(op, deadline)
            msg = f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous '{self._settings.run_id}'."
            self._record(message=msg)
            log.debug(msg)
        except RendezvousError as ex:
            msg = f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
        finally:
            self._heartbeat_lock.release()

    def _start_heartbeats(self) -> None:
        self._keep_alive_timer = _PeriodicTimer(self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self))
        self._keep_alive_timer.set_name(f'RendezvousKeepAliveTimer_{self._this_node.local_id}')
        self._keep_alive_timer.start()

    def _stop_heartbeats(self) -> None:
        if self._keep_alive_timer is None:
            return
        self._keep_alive_timer.cancel()

    def _get_world(self) -> Tuple[int, int]:
        state = self._state_holder.state
        return (state.participants[self._this_node], len(state.participants))

    def _get_store(self) -> Store:
        key_prefix = f'torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}'
        return PrefixStore(key_prefix, self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        return time.monotonic() + timeout.total_seconds()
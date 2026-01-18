from __future__ import annotations
import atexit
import time
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast
from pymongo import common, periodic_executor
from pymongo._csot import MovingMinimum
from pymongo.errors import NotPrimaryError, OperationFailure, _OperationCancelled
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.periodic_executor import _shutdown_executors
from pymongo.pool import _is_faas
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
from pymongo.srv_resolver import _SrvResolver
class MonitorBase:

    def __init__(self, topology: Topology, name: str, interval: int, min_interval: float):
        """Base class to do periodic work on a background thread.

        The background thread is signaled to stop when the Topology or
        this instance is freed.
        """

        def target() -> bool:
            monitor = self_ref()
            if monitor is None:
                return False
            monitor._run()
            return True
        executor = periodic_executor.PeriodicExecutor(interval=interval, min_interval=min_interval, target=target, name=name)
        self._executor = executor

        def _on_topology_gc(dummy: Optional[Topology]=None) -> None:
            monitor = self_ref()
            if monitor:
                monitor.gc_safe_close()
        self_ref = weakref.ref(self, executor.close)
        self._topology = weakref.proxy(topology, _on_topology_gc)
        _register(self)

    def open(self) -> None:
        """Start monitoring, or restart after a fork.

        Multiple calls have no effect.
        """
        self._executor.open()

    def gc_safe_close(self) -> None:
        """GC safe close."""
        self._executor.close()

    def close(self) -> None:
        """Close and stop monitoring.

        open() restarts the monitor after closing.
        """
        self.gc_safe_close()

    def join(self, timeout: Optional[int]=None) -> None:
        """Wait for the monitor to stop."""
        self._executor.join(timeout)

    def request_check(self) -> None:
        """If the monitor is sleeping, wake it soon."""
        self._executor.wake()
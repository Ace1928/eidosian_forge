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
def _check_once(self) -> ServerDescription:
    """A single attempt to call hello.

        Returns a ServerDescription, or raises an exception.
        """
    address = self._server_description.address
    if self._publish:
        assert self._listeners is not None
        sd = self._server_description
        awaited = bool(self._pool.conns and self._stream and sd.is_server_type_known and sd.topology_version)
        self._listeners.publish_server_heartbeat_started(address, awaited)
    if self._cancel_context and self._cancel_context.cancelled:
        self._reset_connection()
    with self._pool.checkout() as conn:
        self._cancel_context = conn.cancel_context
        response, round_trip_time = self._check_with_socket(conn)
        if not response.awaitable:
            self._rtt_monitor.add_sample(round_trip_time)
        avg_rtt, min_rtt = self._rtt_monitor.get()
        sd = ServerDescription(address, response, avg_rtt, min_round_trip_time=min_rtt)
        if self._publish:
            assert self._listeners is not None
            self._listeners.publish_server_heartbeat_succeeded(address, round_trip_time, response, response.awaitable)
        return sd
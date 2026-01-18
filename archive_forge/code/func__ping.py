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
def _ping(self) -> float:
    """Run a "hello" command and return the RTT."""
    with self._pool.checkout() as conn:
        if self._executor._stopped:
            raise Exception('_RttMonitor closed')
        start = time.monotonic()
        conn.hello()
        return time.monotonic() - start
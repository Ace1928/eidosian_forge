from __future__ import annotations
import os
import queue
import random
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast
from pymongo import _csot, common, helpers, periodic_executor
from pymongo.client_session import _ServerSession, _ServerSessionPool
from pymongo.errors import (
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.monitor import SrvMonitor
from pymongo.pool import Pool, PoolOptions
from pymongo.server import Server
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import (
from pymongo.topology_description import (
def _select_servers_loop(self, selector: Callable[[Selection], Selection], timeout: float, address: Optional[_Address]) -> list[ServerDescription]:
    """select_servers() guts. Hold the lock when calling this."""
    now = time.monotonic()
    end_time = now + timeout
    server_descriptions = self._description.apply_selector(selector, address, custom_selector=self._settings.server_selector)
    while not server_descriptions:
        if timeout == 0 or now > end_time:
            raise ServerSelectionTimeoutError(f'{self._error_message(selector)}, Timeout: {timeout}s, Topology Description: {self.description!r}')
        self._ensure_opened()
        self._request_check_all()
        self._condition.wait(common.MIN_HEARTBEAT_INTERVAL)
        self._description.check_compatible()
        now = time.monotonic()
        server_descriptions = self._description.apply_selector(selector, address, custom_selector=self._settings.server_selector)
    self._description.check_compatible()
    return server_descriptions
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
def _ensure_opened(self) -> None:
    """Start monitors, or restart after a fork.

        Hold the lock when calling this.
        """
    if self._closed:
        raise InvalidOperation('Cannot use MongoClient after close')
    if not self._opened:
        self._opened = True
        self._update_servers()
        if self._publish_tp or self._publish_server:
            self.__events_executor.open()
        if self._srv_monitor and self.description.topology_type in SRV_POLLING_TOPOLOGIES:
            self._srv_monitor.open()
        if self._settings.load_balanced:
            self._process_change(ServerDescription(self._seed_addresses[0], Hello({'ok': 1, 'serviceId': self._topology_id, 'maxWireVersion': 13})))
    for server in self._servers.values():
        server.open()
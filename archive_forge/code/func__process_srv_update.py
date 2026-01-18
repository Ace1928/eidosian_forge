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
def _process_srv_update(self, seedlist: list[tuple[str, Any]]) -> None:
    """Process a new seedlist on an opened topology.
        Hold the lock when calling this.
        """
    td_old = self._description
    if td_old.topology_type not in SRV_POLLING_TOPOLOGIES:
        return
    self._description = _updated_topology_description_srv_polling(self._description, seedlist)
    self._update_servers()
    if self._publish_tp:
        assert self._events is not None
        self._events.put((self._listeners.publish_topology_description_changed, (td_old, self._description, self._topology_id)))
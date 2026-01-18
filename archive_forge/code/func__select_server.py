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
def _select_server(self, selector: Callable[[Selection], Selection], server_selection_timeout: Optional[float]=None, address: Optional[_Address]=None) -> Server:
    servers = self.select_servers(selector, server_selection_timeout, address)
    if len(servers) == 1:
        return servers[0]
    server1, server2 = random.sample(servers, 2)
    if server1.pool.operation_count <= server2.pool.operation_count:
        return server1
    else:
        return server2
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
def _create_pool_for_monitor(self, address: _Address) -> Pool:
    options = self._settings.pool_options
    monitor_pool_options = PoolOptions(connect_timeout=options.connect_timeout, socket_timeout=options.connect_timeout, ssl_context=options._ssl_context, tls_allow_invalid_hostnames=options.tls_allow_invalid_hostnames, event_listeners=options._event_listeners, appname=options.appname, driver=options.driver, pause_enabled=False, server_api=options.server_api)
    return self._settings.pool_class(address, monitor_pool_options, handshake=False)
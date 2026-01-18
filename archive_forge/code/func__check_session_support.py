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
def _check_session_support(self) -> float:
    """Internal check for session support on clusters."""
    if self._settings.load_balanced:
        return float('inf')
    session_timeout = self._description.logical_session_timeout_minutes
    if session_timeout is None:
        if self._description.topology_type == TOPOLOGY_TYPE.Single:
            if not self._description.has_known_servers:
                self._select_servers_loop(any_server_selector, self.get_server_selection_timeout(), None)
        elif not self._description.readable_servers:
            self._select_servers_loop(readable_server_selector, self.get_server_selection_timeout(), None)
        session_timeout = self._description.logical_session_timeout_minutes
        if session_timeout is None:
            raise ConfigurationError('Sessions are not supported by this MongoDB deployment')
    return session_timeout
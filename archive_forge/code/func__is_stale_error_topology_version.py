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
def _is_stale_error_topology_version(current_tv: Optional[Mapping[str, Any]], error_tv: Optional[Mapping[str, Any]]) -> bool:
    """Return True if the error's topologyVersion is <= current."""
    if current_tv is None or error_tv is None:
        return False
    if current_tv['processId'] != error_tv['processId']:
        return False
    return current_tv['counter'] >= error_tv['counter']
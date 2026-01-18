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
def _is_stale_server_description(current_sd: ServerDescription, new_sd: ServerDescription) -> bool:
    """Return True if the new topologyVersion is < current."""
    current_tv, new_tv = (current_sd.topology_version, new_sd.topology_version)
    if current_tv is None or new_tv is None:
        return False
    if current_tv['processId'] != new_tv['processId']:
        return False
    return current_tv['counter'] > new_tv['counter']
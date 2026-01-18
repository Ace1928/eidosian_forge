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
def _receive_cluster_time_no_lock(self, cluster_time: Optional[Mapping[str, Any]]) -> None:
    if cluster_time:
        if not self._max_cluster_time or cluster_time['clusterTime'] > self._max_cluster_time['clusterTime']:
            self._max_cluster_time = cluster_time
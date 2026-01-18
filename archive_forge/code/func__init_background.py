from __future__ import annotations
import contextlib
import os
import weakref
from collections import defaultdict
from typing import (
import bson
from bson.codec_options import DEFAULT_CODEC_OPTIONS, TypeRegistry
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import (
from pymongo.change_stream import ChangeStream, ClusterChangeStream
from pymongo.client_options import ClientOptions
from pymongo.client_session import _EmptyServerSession
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.lock import _HAS_REGISTER_AT_FORK, _create_lock, _release_locks
from pymongo.pool import ConnectionClosedReason
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_selectors import writable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.settings import TopologySettings
from pymongo.topology import Topology, _ErrorContext
from pymongo.topology_description import TOPOLOGY_TYPE, TopologyDescription
from pymongo.typings import (
from pymongo.uri_parser import (
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern
def _init_background(self) -> None:
    self._topology = Topology(self._topology_settings)

    def target() -> bool:
        client = self_ref()
        if client is None:
            return False
        MongoClient._process_periodic_tasks(client)
        return True
    executor = periodic_executor.PeriodicExecutor(interval=common.KILL_CURSOR_FREQUENCY, min_interval=common.MIN_HEARTBEAT_INTERVAL, target=target, name='pymongo_kill_cursors_thread')
    self_ref: Any = weakref.ref(self, executor.close)
    self._kill_cursors_executor = executor
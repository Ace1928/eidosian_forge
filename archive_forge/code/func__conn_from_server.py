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
@contextlib.contextmanager
def _conn_from_server(self, read_preference: _ServerMode, server: Server, session: Optional[ClientSession]) -> Iterator[tuple[Connection, _ServerMode]]:
    assert read_preference is not None, 'read_preference must not be None'
    topology = self._get_topology()
    single = topology.description.topology_type == TOPOLOGY_TYPE.Single
    with self._checkout(server, session) as conn:
        if single:
            if conn.is_repl and (not (session and session.in_transaction)):
                read_preference = ReadPreference.PRIMARY_PREFERRED
            elif conn.is_standalone:
                read_preference = ReadPreference.PRIMARY
        yield (conn, read_preference)
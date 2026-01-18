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
def _tmp_session(self, session: Optional[client_session.ClientSession], close: bool=True) -> Generator[Optional[client_session.ClientSession], None, None]:
    """If provided session is None, lend a temporary session."""
    if session is not None:
        if not isinstance(session, client_session.ClientSession):
            raise ValueError("'session' argument must be a ClientSession or None.")
        yield session
        return
    s = self._ensure_session(session)
    if s:
        try:
            yield s
        except Exception as exc:
            if isinstance(exc, ConnectionFailure):
                s._server_session.mark_dirty()
            s.end_session()
            raise
        finally:
            if close:
                s.end_session()
    else:
        yield None
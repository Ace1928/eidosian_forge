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
def _retryable_read(self, func: _ReadCall[T], read_pref: _ServerMode, session: Optional[ClientSession], address: Optional[_Address]=None, retryable: bool=True) -> T:
    """Execute an operation with consecutive retries if possible

        Returns func()'s return value on success. On error retries the same
        command.

        Re-raises any exception thrown by func().

          - `func`: Read call we want to execute
          - `read_pref`: Desired topology of read operation
          - `session`: Client session we should use to execute operation
          - `address`: Optional address when sending a message, defaults to None
          - `retryable`: if we should attempt retries
            (may not always be supported even if supplied), defaults to False
        """
    retryable = bool(retryable and self.options.retry_reads and (not (session and session.in_transaction)))
    return self._retry_internal(func, session, None, is_read=True, address=address, read_pref=read_pref, retryable=retryable)
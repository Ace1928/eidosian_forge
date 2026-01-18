from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
def _perished(self, conn: Connection) -> bool:
    """Return True and close the connection if it is "perished".

        This side-effecty function checks if this socket has been idle for
        for longer than the max idle time, or if the socket has been closed by
        some external network error, or if the socket's generation is outdated.

        Checking sockets lets us avoid seeing *some*
        :class:`~pymongo.errors.AutoReconnect` exceptions on server
        hiccups, etc. We only check if the socket was closed by an external
        error if it has been > 1 second since the socket was checked into the
        pool, to keep performance reasonable - we can't avoid AutoReconnects
        completely anyway.
        """
    idle_time_seconds = conn.idle_time_seconds()
    if self.opts.max_idle_time_seconds is not None and idle_time_seconds > self.opts.max_idle_time_seconds:
        conn.close_conn(ConnectionClosedReason.IDLE)
        return True
    if self._check_interval_seconds is not None and (self._check_interval_seconds == 0 or idle_time_seconds > self._check_interval_seconds):
        if conn.conn_closed():
            conn.close_conn(ConnectionClosedReason.ERROR)
            return True
    if self.stale_generation(conn.generation, conn.service_id):
        conn.close_conn(ConnectionClosedReason.STALE)
        return True
    return False
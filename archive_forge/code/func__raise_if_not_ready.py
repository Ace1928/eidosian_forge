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
def _raise_if_not_ready(self, emit_event: bool) -> None:
    if self.state != PoolState.READY:
        if self.enabled_for_cmap and emit_event:
            assert self.opts._event_listeners is not None
            self.opts._event_listeners.publish_connection_check_out_failed(self.address, ConnectionCheckOutFailedReason.CONN_ERROR)
        details = _get_timeout_details(self.opts)
        _raise_connection_failure(self.address, AutoReconnect('connection pool paused'), timeout_details=details)
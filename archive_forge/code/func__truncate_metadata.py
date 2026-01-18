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
def _truncate_metadata(metadata: MutableMapping[str, Any]) -> None:
    """Perform metadata truncation."""
    if len(bson.encode(metadata)) <= _MAX_METADATA_SIZE:
        return
    env_name = metadata.get('env', {}).get('name')
    if env_name:
        metadata['env'] = {'name': env_name}
    if len(bson.encode(metadata)) <= _MAX_METADATA_SIZE:
        return
    os_type = metadata.get('os', {}).get('type')
    if os_type:
        metadata['os'] = {'type': os_type}
    if len(bson.encode(metadata)) <= _MAX_METADATA_SIZE:
        return
    metadata.pop('env', None)
    encoded_size = len(bson.encode(metadata))
    if encoded_size <= _MAX_METADATA_SIZE:
        return
    overflow = encoded_size - _MAX_METADATA_SIZE
    plat = metadata.get('platform', '')
    if plat:
        plat = plat[:-overflow]
    if plat:
        metadata['platform'] = plat
    else:
        metadata.pop('platform', None)
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
def _add_retryable_write_error(exc: PyMongoError, max_wire_version: int) -> None:
    doc = _retryable_error_doc(exc)
    if doc:
        code = doc.get('code', 0)
        if code == 20 and str(exc).startswith('Transaction numbers'):
            errmsg = 'This MongoDB deployment does not support retryable writes. Please add retryWrites=false to your connection string.'
            raise OperationFailure(errmsg, code, exc.details)
        if max_wire_version >= 9:
            for label in doc.get('errorLabels', []):
                exc._add_error_label(label)
        elif code in helpers._RETRYABLE_ERROR_CODES:
            exc._add_error_label('RetryableWriteError')
    if isinstance(exc, ConnectionFailure) and (not isinstance(exc, (NotPrimaryError, WaitQueueTimeoutError))):
        exc._add_error_label('RetryableWriteError')
from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
class _Transaction:
    """Internal class to hold transaction information in a ClientSession."""

    def __init__(self, opts: Optional[TransactionOptions], client: MongoClient):
        self.opts = opts
        self.state = _TxnState.NONE
        self.sharded = False
        self.pinned_address: Optional[_Address] = None
        self.conn_mgr: Optional[_ConnectionManager] = None
        self.recovery_token = None
        self.attempt = 0
        self.client = client

    def active(self) -> bool:
        return self.state in (_TxnState.STARTING, _TxnState.IN_PROGRESS)

    def starting(self) -> bool:
        return self.state == _TxnState.STARTING

    @property
    def pinned_conn(self) -> Optional[Connection]:
        if self.active() and self.conn_mgr:
            return self.conn_mgr.conn
        return None

    def pin(self, server: Server, conn: Connection) -> None:
        self.sharded = True
        self.pinned_address = server.description.address
        if server.description.server_type == SERVER_TYPE.LoadBalancer:
            conn.pin_txn()
            self.conn_mgr = _ConnectionManager(conn, False)

    def unpin(self) -> None:
        self.pinned_address = None
        if self.conn_mgr:
            self.conn_mgr.close()
        self.conn_mgr = None

    def reset(self) -> None:
        self.unpin()
        self.state = _TxnState.NONE
        self.sharded = False
        self.recovery_token = None
        self.attempt = 0

    def __del__(self) -> None:
        if self.conn_mgr:
            self.client._close_cursor_soon(0, None, self.conn_mgr)
            self.conn_mgr = None
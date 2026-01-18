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
class _ServerSession:

    def __init__(self, generation: int):
        self.session_id = {'id': Binary(uuid.uuid4().bytes, 4)}
        self.last_use = time.monotonic()
        self._transaction_id = 0
        self.dirty = False
        self.generation = generation

    def mark_dirty(self) -> None:
        """Mark this session as dirty.

        A server session is marked dirty when a command fails with a network
        error. Dirty sessions are later discarded from the server session pool.
        """
        self.dirty = True

    def timed_out(self, session_timeout_minutes: float) -> bool:
        idle_seconds = time.monotonic() - self.last_use
        return idle_seconds > (session_timeout_minutes - 1) * 60

    @property
    def transaction_id(self) -> Int64:
        """Positive 64-bit integer."""
        return Int64(self._transaction_id)

    def inc_transaction_id(self) -> None:
        self._transaction_id += 1
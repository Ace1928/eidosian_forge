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
class _EmptyServerSession:
    __slots__ = ('dirty', 'started_retryable_write')

    def __init__(self) -> None:
        self.dirty = False
        self.started_retryable_write = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def inc_transaction_id(self) -> None:
        self.started_retryable_write = True
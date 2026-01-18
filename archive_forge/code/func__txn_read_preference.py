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
def _txn_read_preference(self) -> Optional[_ServerMode]:
    """Return read preference of this transaction or None."""
    if self.in_transaction:
        assert self._transaction.opts
        return self._transaction.opts.read_preference
    return None
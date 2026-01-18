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
def _validate_session_write_concern(session: Optional[ClientSession], write_concern: Optional[WriteConcern]) -> Optional[ClientSession]:
    """Validate that an explicit session is not used with an unack'ed write.

    Returns the session to use for the next operation.
    """
    if session:
        if write_concern is not None and (not write_concern.acknowledged):
            if session._implicit:
                return None
            else:
                raise ConfigurationError(f'Explicit sessions are incompatible with unacknowledged write concern: {write_concern!r}')
    return session
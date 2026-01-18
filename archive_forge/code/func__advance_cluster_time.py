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
def _advance_cluster_time(self, cluster_time: Optional[Mapping[str, Any]]) -> None:
    """Internal cluster time helper."""
    if self._cluster_time is None:
        self._cluster_time = cluster_time
    elif cluster_time is not None:
        if cluster_time['clusterTime'] > self._cluster_time['clusterTime']:
            self._cluster_time = cluster_time
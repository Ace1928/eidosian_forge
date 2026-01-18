from __future__ import annotations
import copy
import warnings
from collections import deque
from typing import (
from bson import RE_TYPE, _convert_raw_document_lists_to_streams
from bson.code import Code
from bson.son import SON
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import (
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.lock import _create_lock
from pymongo.message import (
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _CollationIn, _DocumentOut, _DocumentType
class _ConnectionManager:
    """Used with exhaust cursors to ensure the connection is returned."""

    def __init__(self, conn: Connection, more_to_come: bool):
        self.conn: Optional[Connection] = conn
        self.more_to_come = more_to_come
        self.lock = _create_lock()

    def update_exhaust(self, more_to_come: bool) -> None:
        self.more_to_come = more_to_come

    def close(self) -> None:
        """Return this instance's connection to the connection pool."""
        if self.conn:
            self.conn.unpin()
            self.conn = None
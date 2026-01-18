from __future__ import annotations
import datetime
import io
import math
import os
from typing import Any, Iterable, Mapping, NoReturn, Optional
from bson.binary import Binary
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.son import SON
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import (
from pymongo.read_preferences import ReadPreference
def _next_with_retry(self) -> Mapping[str, Any]:
    """Return the next chunk and retry once on CursorNotFound.

        We retry on CursorNotFound to maintain backwards compatibility in
        cases where two calls to read occur more than 10 minutes apart (the
        server's default cursor timeout).
        """
    if self._cursor is None:
        self._create_cursor()
        assert self._cursor is not None
    try:
        return self._cursor.next()
    except CursorNotFound:
        self._cursor.close()
        self._create_cursor()
        return self._cursor.next()
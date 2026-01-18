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
def __flush_data(self, data: Any) -> None:
    """Flush `data` to a chunk."""
    self.__ensure_indexes()
    if not data:
        return
    assert len(data) <= self.chunk_size
    chunk = {'files_id': self._file['_id'], 'n': self._chunk_number, 'data': Binary(data)}
    try:
        self._chunks.insert_one(chunk, session=self._session)
    except DuplicateKeyError:
        self._raise_file_exists(self._file['_id'])
    self._chunk_number += 1
    self._position += len(data)
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
def _create_cursor(self) -> None:
    filter = {'files_id': self._id}
    if self._next_chunk > 0:
        filter['n'] = {'$gte': self._next_chunk}
    _disallow_transactions(self._session)
    self._cursor = self._chunks.find(filter, sort=[('n', 1)], session=self._session)
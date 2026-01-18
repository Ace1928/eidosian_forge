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
def __create_index(self, collection: Collection, index_key: Any, unique: bool) -> None:
    doc = collection.find_one(projection={'_id': 1}, session=self._session)
    if doc is None:
        try:
            index_keys = [index_spec['key'] for index_spec in collection.list_indexes(session=self._session)]
        except OperationFailure:
            index_keys = []
        if index_key not in index_keys:
            collection.create_index(index_key.items(), unique=unique, session=self._session)
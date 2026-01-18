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
class GridOutIterator:

    def __init__(self, grid_out: GridOut, chunks: Collection, session: ClientSession):
        self.__chunk_iter = _GridOutChunkIterator(grid_out, chunks, session, 0)

    def __iter__(self) -> GridOutIterator:
        return self

    def next(self) -> bytes:
        chunk = self.__chunk_iter.next()
        return bytes(chunk['data'])
    __next__ = next
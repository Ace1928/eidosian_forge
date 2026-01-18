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
def _read_size_or_line(self, size: int=-1, line: bool=False) -> bytes:
    """Internal read() and readline() helper."""
    self._ensure_file()
    remainder = int(self.length) - self.__position
    if size < 0 or size > remainder:
        size = remainder
    if size == 0:
        return EMPTY
    received = 0
    data = []
    while received < size:
        needed = size - received
        if self.__buffer:
            buf = self.__buffer
            chunk_start = self.__buffer_pos
            chunk_data = memoryview(buf)[self.__buffer_pos:]
            self.__buffer = EMPTY
            self.__buffer_pos = 0
            self.__position += len(chunk_data)
        else:
            buf = self.readchunk()
            chunk_start = 0
            chunk_data = memoryview(buf)
        if line:
            pos = buf.find(NEWLN, chunk_start, chunk_start + needed) - chunk_start
            if pos >= 0:
                size = received + pos + 1
                needed = pos + 1
        if len(chunk_data) > needed:
            data.append(chunk_data[:needed])
            self.__buffer = buf
            self.__buffer_pos = chunk_start + needed
            self.__position -= len(self.__buffer) - self.__buffer_pos
        else:
            data.append(chunk_data)
        received += len(chunk_data)
    if size == remainder and self.__chunk_iter:
        try:
            self.__chunk_iter.next()
        except StopIteration:
            pass
    return b''.join(data)
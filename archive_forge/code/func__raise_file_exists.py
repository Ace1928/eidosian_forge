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
def _raise_file_exists(self, file_id: Any) -> NoReturn:
    """Raise a FileExists exception for the given file_id."""
    raise FileExists('file with _id %r already exists' % file_id)
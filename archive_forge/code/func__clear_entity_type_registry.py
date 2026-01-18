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
def _clear_entity_type_registry(entity: Any, **kwargs: Any) -> Any:
    """Clear the given database/collection object's type registry."""
    codecopts = entity.codec_options.with_options(type_registry=None)
    return entity.with_options(codec_options=codecopts, **kwargs)
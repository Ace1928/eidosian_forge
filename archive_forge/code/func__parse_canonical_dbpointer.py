from __future__ import annotations
import base64
import datetime
import json
import math
import re
import uuid
from typing import (
from bson.binary import ALL_UUID_SUBTYPES, UUID_SUBTYPE, Binary, UuidRepresentation
from bson.code import Code
from bson.codec_options import CodecOptions, DatetimeConversion
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def _parse_canonical_dbpointer(doc: Any) -> Any:
    """Decode a JSON (deprecated) DBPointer to bson.dbref.DBRef."""
    dbref = doc['$dbPointer']
    if len(doc) != 1:
        raise TypeError(f'Bad $dbPointer, extra field(s): {doc}')
    if isinstance(dbref, DBRef):
        dbref_doc = dbref.as_doc()
        if dbref.database is not None:
            raise TypeError(f'Bad $dbPointer, extra field $db: {dbref_doc}')
        if not isinstance(dbref.id, ObjectId):
            raise TypeError(f'Bad $dbPointer, $id must be an ObjectId: {dbref_doc}')
        if len(dbref_doc) != 2:
            raise TypeError(f'Bad $dbPointer, extra field(s) in DBRef: {dbref_doc}')
        return dbref
    else:
        raise TypeError(f'Bad $dbPointer, expected a DBRef: {doc}')
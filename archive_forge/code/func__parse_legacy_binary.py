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
def _parse_legacy_binary(doc: Any, json_options: JSONOptions) -> Union[Binary, uuid.UUID]:
    if isinstance(doc['$type'], int):
        doc['$type'] = '%02x' % doc['$type']
    subtype = int(doc['$type'], 16)
    if subtype >= 4294967168:
        subtype = int(doc['$type'][6:], 16)
    data = base64.b64decode(doc['$binary'].encode())
    return _binary_or_uuid(data, subtype, json_options)
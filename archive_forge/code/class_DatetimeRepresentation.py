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
class DatetimeRepresentation:
    LEGACY = 0
    'Legacy MongoDB Extended JSON datetime representation.\n\n    :class:`datetime.datetime` instances will be encoded to JSON in the\n    format `{"$date": <dateAsMilliseconds>}`, where `dateAsMilliseconds` is\n    a 64-bit signed integer giving the number of milliseconds since the Unix\n    epoch UTC. This was the default encoding before PyMongo version 3.4.\n\n    .. versionadded:: 3.4\n    '
    NUMBERLONG = 1
    'NumberLong datetime representation.\n\n    :class:`datetime.datetime` instances will be encoded to JSON in the\n    format `{"$date": {"$numberLong": "<dateAsMilliseconds>"}}`,\n    where `dateAsMilliseconds` is the string representation of a 64-bit signed\n    integer giving the number of milliseconds since the Unix epoch UTC.\n\n    .. versionadded:: 3.4\n    '
    ISO8601 = 2
    'ISO-8601 datetime representation.\n\n    :class:`datetime.datetime` instances greater than or equal to the Unix\n    epoch UTC will be encoded to JSON in the format `{"$date": "<ISO-8601>"}`.\n    :class:`datetime.datetime` instances before the Unix epoch UTC will be\n    encoded as if the datetime representation is\n    :const:`~DatetimeRepresentation.NUMBERLONG`.\n\n    .. versionadded:: 3.4\n    '
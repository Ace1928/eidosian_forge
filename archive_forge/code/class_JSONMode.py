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
class JSONMode:
    LEGACY = 0
    "Legacy Extended JSON representation.\n\n    In this mode, :func:`~bson.json_util.dumps` produces PyMongo's legacy\n    non-standard JSON output. Consider using\n    :const:`~bson.json_util.JSONMode.RELAXED` or\n    :const:`~bson.json_util.JSONMode.CANONICAL` instead.\n\n    .. versionadded:: 3.5\n    "
    RELAXED = 1
    'Relaxed Extended JSON representation.\n\n    In this mode, :func:`~bson.json_util.dumps` produces Relaxed Extended JSON,\n    a mostly JSON-like format. Consider using this for things like a web API,\n    where one is sending a document (or a projection of a document) that only\n    uses ordinary JSON type primitives. In particular, the ``int``,\n    :class:`~bson.int64.Int64`, and ``float`` numeric types are represented in\n    the native JSON number format. This output is also the most human readable\n    and is useful for debugging and documentation.\n\n    .. seealso:: The specification for Relaxed `Extended JSON`_.\n\n    .. versionadded:: 3.5\n    '
    CANONICAL = 2
    'Canonical Extended JSON representation.\n\n    In this mode, :func:`~bson.json_util.dumps` produces Canonical Extended\n    JSON, a type preserving format. Consider using this for things like\n    testing, where one has to precisely specify expected types in JSON. In\n    particular, the ``int``, :class:`~bson.int64.Int64`, and ``float`` numeric\n    types are encoded with type wrappers.\n\n    .. seealso:: The specification for Canonical `Extended JSON`_.\n\n    .. versionadded:: 3.5\n    '
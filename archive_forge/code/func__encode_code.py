from __future__ import annotations
import datetime
import itertools
import os
import re
import struct
import sys
import uuid
from codecs import utf_8_decode as _utf_8_decode
from codecs import utf_8_encode as _utf_8_encode
from collections import abc as _abc
from typing import (
from bson.binary import (
from bson.code import Code
from bson.codec_options import (
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.errors import InvalidBSON, InvalidDocument, InvalidStringData
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def _encode_code(name: bytes, value: Code, dummy: Any, opts: CodecOptions[Any]) -> bytes:
    """Encode bson.code.Code."""
    cstring = _make_c_string(value)
    cstrlen = len(cstring)
    if value.scope is None:
        return b'\r' + name + _PACK_INT(cstrlen) + cstring
    scope = _dict_to_bson(value.scope, False, opts, False)
    full_length = _PACK_INT(8 + cstrlen + len(scope))
    return b'\x0f' + name + full_length + _PACK_INT(cstrlen) + cstring + scope
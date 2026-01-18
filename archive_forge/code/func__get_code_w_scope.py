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
def _get_code_w_scope(data: Any, view: Any, position: int, _obj_end: int, opts: CodecOptions[Any], element_name: str) -> Tuple[Code, int]:
    """Decode a BSON code_w_scope to bson.code.Code."""
    code_end = position + _UNPACK_INT_FROM(data, position)[0]
    code, position = _get_string(data, view, position + 4, code_end, opts, element_name)
    scope, position = _get_object(data, view, position, code_end, opts, element_name)
    if position != code_end:
        raise InvalidBSON('scope outside of javascript code boundaries')
    return (Code(code, scope), position)
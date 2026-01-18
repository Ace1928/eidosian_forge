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
def _encode_mapping(name: bytes, value: Any, check_keys: bool, opts: CodecOptions[Any]) -> bytes:
    """Encode a mapping type."""
    if _raw_document_class(value):
        return b'\x03' + name + cast(bytes, value.raw)
    data = b''.join([_element_to_bson(key, val, check_keys, opts) for key, val in value.items()])
    return b'\x03' + name + _PACK_INT(len(data) + 5) + data + b'\x00'
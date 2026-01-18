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
def _encode_dbref(name: bytes, value: DBRef, check_keys: bool, opts: CodecOptions[Any]) -> bytes:
    """Encode bson.dbref.DBRef."""
    buf = bytearray(b'\x03' + name + b'\x00\x00\x00\x00')
    begin = len(buf) - 4
    buf += _name_value_to_bson(b'$ref\x00', value.collection, check_keys, opts)
    buf += _name_value_to_bson(b'$id\x00', value.id, check_keys, opts)
    if value.database is not None:
        buf += _name_value_to_bson(b'$db\x00', value.database, check_keys, opts)
    for key, val in value._DBRef__kwargs.items():
        buf += _element_to_bson(key, val, check_keys, opts)
    buf += b'\x00'
    buf[begin:begin + 4] = _PACK_INT(len(buf) - begin)
    return bytes(buf)
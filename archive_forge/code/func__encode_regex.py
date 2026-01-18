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
def _encode_regex(name: bytes, value: Regex[Any], dummy0: Any, dummy1: Any) -> bytes:
    """Encode a python regex or bson.regex.Regex."""
    flags = value.flags
    if flags == re.UNICODE:
        return b'\x0b' + name + _make_c_string_check(value.pattern) + b'u\x00'
    elif flags == 0:
        return b'\x0b' + name + _make_c_string_check(value.pattern) + b'\x00'
    else:
        sflags = b''
        if flags & re.IGNORECASE:
            sflags += b'i'
        if flags & re.LOCALE:
            sflags += b'l'
        if flags & re.MULTILINE:
            sflags += b'm'
        if flags & re.DOTALL:
            sflags += b's'
        if flags & re.UNICODE:
            sflags += b'u'
        if flags & re.VERBOSE:
            sflags += b'x'
        sflags += b'\x00'
        return b'\x0b' + name + _make_c_string_check(value.pattern) + sflags
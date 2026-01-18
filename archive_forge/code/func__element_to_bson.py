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
def _element_to_bson(key: Any, value: Any, check_keys: bool, opts: CodecOptions[Any]) -> bytes:
    """Encode a single key, value pair."""
    if not isinstance(key, str):
        raise InvalidDocument(f'documents must have only string keys, key was {key!r}')
    if check_keys:
        if key.startswith('$'):
            raise InvalidDocument(f"key {key!r} must not start with '$'")
        if '.' in key:
            raise InvalidDocument(f"key {key!r} must not contain '.'")
    name = _make_name(key)
    return _name_value_to_bson(name, value, check_keys, opts)
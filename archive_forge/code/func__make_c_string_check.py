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
def _make_c_string_check(string: Union[str, bytes]) -> bytes:
    """Make a 'C' string, checking for embedded NUL characters."""
    if isinstance(string, bytes):
        if b'\x00' in string:
            raise InvalidDocument('BSON keys / regex patterns must not contain a NUL character')
        try:
            _utf_8_decode(string, None, True)
            return string + b'\x00'
        except UnicodeError:
            raise InvalidStringData('strings in documents must be valid UTF-8: %r' % string) from None
    else:
        if '\x00' in string:
            raise InvalidDocument('BSON keys / regex patterns must not contain a NUL character')
        return _utf_8_encode(string)[0] + b'\x00'
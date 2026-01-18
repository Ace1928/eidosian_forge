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
def _array_of_documents_to_buffer(view: memoryview) -> bytes:
    position = 0
    _, end = _get_object_size(view, position, len(view))
    position += 4
    buffers: list[memoryview] = []
    append = buffers.append
    while position < end - 1:
        while view[position] != 0:
            position += 1
        position += 1
        obj_size, _ = _get_object_size(view, position, end)
        append(view[position:position + obj_size])
        position += obj_size
    if position != end:
        raise InvalidBSON('bad object or element length')
    return b''.join(buffers)
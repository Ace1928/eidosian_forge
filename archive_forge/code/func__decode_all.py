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
def _decode_all(data: _ReadableBuffer, opts: CodecOptions[_DocumentType]) -> list[_DocumentType]:
    """Decode a BSON data to multiple documents."""
    data, view = get_data_and_view(data)
    data_len = len(data)
    docs: list[_DocumentType] = []
    position = 0
    end = data_len - 1
    use_raw = _raw_document_class(opts.document_class)
    try:
        while position < end:
            obj_size = _UNPACK_INT_FROM(data, position)[0]
            if data_len - position < obj_size:
                raise InvalidBSON('invalid object size')
            obj_end = position + obj_size - 1
            if data[obj_end] != 0:
                raise InvalidBSON('bad eoo')
            if use_raw:
                docs.append(opts.document_class(data[position:obj_end + 1], opts))
            else:
                docs.append(_elements_to_dict(data, view, position + 4, obj_end, opts))
            position += obj_size
        return docs
    except InvalidBSON:
        raise
    except Exception:
        _, exc_value, exc_tb = sys.exc_info()
        raise InvalidBSON(str(exc_value)).with_traceback(exc_tb) from None
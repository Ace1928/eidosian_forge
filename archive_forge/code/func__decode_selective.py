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
def _decode_selective(rawdoc: Any, fields: Any, codec_options: CodecOptions[_DocumentType]) -> _DocumentType:
    if _raw_document_class(codec_options.document_class):
        doc: _DocumentType = {}
    else:
        doc = codec_options.document_class()
    for key, value in rawdoc.items():
        if key in fields:
            if fields[key] == 1:
                doc[key] = _bson_to_dict(rawdoc.raw, codec_options)[key]
            else:
                doc[key] = _decode_selective(value, fields[key], codec_options)
        else:
            doc[key] = value
    return doc
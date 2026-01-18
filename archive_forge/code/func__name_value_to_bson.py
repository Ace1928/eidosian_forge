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
def _name_value_to_bson(name: bytes, value: Any, check_keys: bool, opts: CodecOptions[Any], in_custom_call: bool=False, in_fallback_call: bool=False) -> bytes:
    """Encode a single name, value pair."""
    was_integer_overflow = False
    try:
        return _ENCODERS[type(value)](name, value, check_keys, opts)
    except KeyError:
        pass
    except OverflowError:
        if not isinstance(value, int):
            raise
        was_integer_overflow = True
    marker = getattr(value, '_type_marker', None)
    if isinstance(marker, int) and marker in _MARKERS:
        func = _MARKERS[marker]
        _ENCODERS[type(value)] = func
        return func(name, value, check_keys, opts)
    if not in_custom_call and opts.type_registry._encoder_map:
        custom_encoder = opts.type_registry._encoder_map.get(type(value))
        if custom_encoder is not None:
            return _name_value_to_bson(name, custom_encoder(value), check_keys, opts, in_custom_call=True)
    for base in _BUILT_IN_TYPES:
        if not was_integer_overflow and isinstance(value, base):
            func = _ENCODERS[base]
            _ENCODERS[type(value)] = func
            return func(name, value, check_keys, opts)
    fallback_encoder = opts.type_registry._fallback_encoder
    if not in_fallback_call and fallback_encoder is not None:
        return _name_value_to_bson(name, fallback_encoder(value), check_keys, opts, in_fallback_call=True)
    if was_integer_overflow:
        raise OverflowError('BSON can only handle up to 8-byte ints')
    raise InvalidDocument(f'cannot encode object: {value!r}, of type: {type(value)!r}')
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
def _decode_all_selective(data: Any, codec_options: CodecOptions[_DocumentType], fields: Any) -> list[_DocumentType]:
    """Decode BSON data to a single document while using user-provided
    custom decoding logic.

    `data` must be a string representing a valid, BSON-encoded document.

    :Parameters:
      - `data`: BSON data
      - `codec_options`: An instance of
        :class:`~bson.codec_options.CodecOptions` with user-specified type
        decoders. If no decoders are found, this method is the same as
        ``decode_all``.
      - `fields`: Map of document namespaces where data that needs
        to be custom decoded lives or None. For example, to custom decode a
        list of objects in 'field1.subfield1', the specified value should be
        ``{'field1': {'subfield1': 1}}``. If ``fields``  is an empty map or
        None, this method is the same as ``decode_all``.

    :Returns:
      - `document_list`: Single-member list containing the decoded document.

    .. versionadded:: 3.8
    """
    if not codec_options.type_registry._decoder_map:
        return decode_all(data, codec_options)
    if not fields:
        return decode_all(data, codec_options.with_options(type_registry=None))
    from bson.raw_bson import RawBSONDocument
    internal_codec_options: CodecOptions[RawBSONDocument] = codec_options.with_options(document_class=RawBSONDocument, type_registry=None)
    _doc = _bson_to_dict(data, internal_codec_options)
    return [_decode_selective(_doc, fields, codec_options)]
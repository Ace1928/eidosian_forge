from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def _batched_write_command_impl(namespace: str, operation: int, command: MutableMapping[str, Any], docs: list[Mapping[str, Any]], opts: CodecOptions, ctx: _BulkWriteContext, buf: _BytesIO) -> tuple[list[Mapping[str, Any]], int]:
    """Create a batched OP_QUERY write command."""
    max_bson_size = ctx.max_bson_size
    max_write_batch_size = ctx.max_write_batch_size
    max_cmd_size = max_bson_size + _COMMAND_OVERHEAD
    max_split_size = ctx.max_split_size
    buf.write(_ZERO_32)
    buf.write(namespace.encode('utf8'))
    buf.write(_ZERO_8)
    buf.write(_SKIPLIM)
    command_start = buf.tell()
    buf.write(encode(command))
    buf.seek(-1, 2)
    buf.truncate()
    try:
        buf.write(_OP_MAP[operation])
    except KeyError:
        raise InvalidOperation('Unknown command') from None
    list_start = buf.tell() - 4
    to_send = []
    idx = 0
    for doc in docs:
        key = str(idx).encode('utf8')
        value = _dict_to_bson(doc, False, opts)
        doc_too_large = len(value) > max_cmd_size
        if doc_too_large:
            write_op = list(_FIELD_MAP.keys())[operation]
            _raise_document_too_large(write_op, len(value), max_bson_size)
        enough_data = idx >= 1 and buf.tell() + len(key) + len(value) >= max_split_size
        enough_documents = idx >= max_write_batch_size
        if enough_data or enough_documents:
            break
        buf.write(_BSONOBJ)
        buf.write(key)
        buf.write(_ZERO_8)
        buf.write(value)
        to_send.append(doc)
        idx += 1
    buf.write(_ZERO_16)
    length = buf.tell()
    buf.seek(list_start)
    buf.write(_pack_int(length - list_start - 1))
    buf.seek(command_start)
    buf.write(_pack_int(length - command_start))
    return (to_send, length)
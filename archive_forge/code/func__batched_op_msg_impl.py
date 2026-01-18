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
def _batched_op_msg_impl(operation: int, command: Mapping[str, Any], docs: list[Mapping[str, Any]], ack: bool, opts: CodecOptions, ctx: _BulkWriteContext, buf: _BytesIO) -> tuple[list[Mapping[str, Any]], int]:
    """Create a batched OP_MSG write."""
    max_bson_size = ctx.max_bson_size
    max_write_batch_size = ctx.max_write_batch_size
    max_message_size = ctx.max_message_size
    flags = b'\x00\x00\x00\x00' if ack else b'\x02\x00\x00\x00'
    buf.write(flags)
    buf.write(b'\x00')
    buf.write(_dict_to_bson(command, False, opts))
    buf.write(b'\x01')
    size_location = buf.tell()
    buf.write(b'\x00\x00\x00\x00')
    try:
        buf.write(_OP_MSG_MAP[operation])
    except KeyError:
        raise InvalidOperation('Unknown command') from None
    to_send = []
    idx = 0
    for doc in docs:
        value = _dict_to_bson(doc, False, opts)
        doc_length = len(value)
        new_message_size = buf.tell() + doc_length
        doc_too_large = idx == 0 and new_message_size > max_message_size
        unacked_doc_too_large = not ack and doc_length > max_bson_size
        if doc_too_large or unacked_doc_too_large:
            write_op = list(_FIELD_MAP.keys())[operation]
            _raise_document_too_large(write_op, len(value), max_bson_size)
        if new_message_size > max_message_size:
            break
        buf.write(value)
        to_send.append(doc)
        idx += 1
        if idx == max_write_batch_size:
            break
    length = buf.tell()
    buf.seek(size_location)
    buf.write(_pack_int(length - size_location))
    return (to_send, length)
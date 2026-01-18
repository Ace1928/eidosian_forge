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
def _batched_op_msg_compressed(operation: int, command: Mapping[str, Any], docs: list[Mapping[str, Any]], ack: bool, opts: CodecOptions, ctx: _BulkWriteContext) -> tuple[int, bytes, list[Mapping[str, Any]]]:
    """Create the next batched insert, update, or delete operation
    with OP_MSG, compressed.
    """
    data, to_send = _encode_batched_op_msg(operation, command, docs, ack, opts, ctx)
    assert ctx.conn.compression_context is not None
    request_id, msg = _compress(2013, data, ctx.conn.compression_context)
    return (request_id, msg, to_send)
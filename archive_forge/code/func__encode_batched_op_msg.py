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
def _encode_batched_op_msg(operation: int, command: Mapping[str, Any], docs: list[Mapping[str, Any]], ack: bool, opts: CodecOptions, ctx: _BulkWriteContext) -> tuple[bytes, list[Mapping[str, Any]]]:
    """Encode the next batched insert, update, or delete operation
    as OP_MSG.
    """
    buf = _BytesIO()
    to_send, _ = _batched_op_msg_impl(operation, command, docs, ack, opts, ctx, buf)
    return (buf.getvalue(), to_send)
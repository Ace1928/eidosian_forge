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
def _gen_get_more_command(cursor_id: Optional[int], coll: str, batch_size: Optional[int], max_await_time_ms: Optional[int], comment: Optional[Any], conn: Connection) -> SON[str, Any]:
    """Generate a getMore command document."""
    cmd: SON[str, Any] = SON([('getMore', cursor_id), ('collection', coll)])
    if batch_size:
        cmd['batchSize'] = batch_size
    if max_await_time_ms is not None:
        cmd['maxTimeMS'] = max_await_time_ms
    if comment is not None and conn.max_wire_version >= 9:
        cmd['comment'] = comment
    return cmd
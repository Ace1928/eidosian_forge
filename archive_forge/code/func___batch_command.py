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
def __batch_command(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]]) -> tuple[MutableMapping[str, Any], list[Mapping[str, Any]]]:
    namespace = self.db_name + '.$cmd'
    msg, to_send = _encode_batched_write_command(namespace, self.op_type, cmd, docs, self.codec, self)
    if not to_send:
        raise InvalidOperation('cannot do an empty bulk write')
    cmd_start = msg.index(b'\x00', 4) + 9
    outgoing = _inflate_bson(memoryview(msg)[cmd_start:], DEFAULT_RAW_BSON_OPTIONS)
    return (outgoing, to_send)
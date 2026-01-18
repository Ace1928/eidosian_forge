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
def _query_impl(options: int, collection_name: str, num_to_skip: int, num_to_return: int, query: Mapping[str, Any], field_selector: Optional[Mapping[str, Any]], opts: CodecOptions) -> tuple[bytes, int]:
    """Get an OP_QUERY message."""
    encoded = _dict_to_bson(query, False, opts)
    if field_selector:
        efs = _dict_to_bson(field_selector, False, opts)
    else:
        efs = b''
    max_bson_size = max(len(encoded), len(efs))
    return (b''.join([_pack_int(options), _make_c_string(collection_name), _pack_int(num_to_skip), _pack_int(num_to_return), encoded, efs]), max_bson_size)
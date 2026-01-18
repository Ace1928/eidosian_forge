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
def _get_more_compressed(collection_name: str, num_to_return: int, cursor_id: int, ctx: Union[SnappyContext, ZlibContext, ZstdContext]) -> tuple[int, bytes]:
    """Internal compressed getMore message helper."""
    return _compress(2005, _get_more_impl(collection_name, num_to_return, cursor_id), ctx)
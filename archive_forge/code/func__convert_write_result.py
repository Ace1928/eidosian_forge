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
def _convert_write_result(operation: str, command: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a legacy write result to write command format."""
    affected = result.get('n', 0)
    res = {'ok': 1, 'n': affected}
    errmsg = result.get('errmsg', result.get('err', ''))
    if errmsg:
        if result.get('wtimeout'):
            res['writeConcernError'] = {'errmsg': errmsg, 'code': 64, 'errInfo': {'wtimeout': True}}
        else:
            error = {'index': 0, 'code': result.get('code', 8), 'errmsg': errmsg}
            if 'errInfo' in result:
                error['errInfo'] = result['errInfo']
            res['writeErrors'] = [error]
            return res
    if operation == 'insert':
        res['n'] = len(command['documents'])
    elif operation == 'update':
        if 'upserted' in result:
            res['upserted'] = [{'index': 0, '_id': result['upserted']}]
        elif result.get('updatedExisting') is False and affected == 1:
            update = command['updates'][0]
            _id = update['u'].get('_id', update['q'].get('_id'))
            res['upserted'] = [{'index': 0, '_id': _id}]
    return res
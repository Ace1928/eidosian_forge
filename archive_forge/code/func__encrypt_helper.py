from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
def _encrypt_helper(self, value: Any, algorithm: str, key_id: Optional[Binary]=None, key_alt_name: Optional[str]=None, query_type: Optional[str]=None, contention_factor: Optional[int]=None, range_opts: Optional[RangeOpts]=None, is_expression: bool=False) -> Any:
    self._check_closed()
    if key_id is not None and (not (isinstance(key_id, Binary) and key_id.subtype == UUID_SUBTYPE)):
        raise TypeError('key_id must be a bson.binary.Binary with subtype 4')
    doc = encode({'v': value}, codec_options=self._codec_options)
    range_opts_bytes = None
    if range_opts:
        range_opts_bytes = encode(range_opts.document, codec_options=self._codec_options)
    with _wrap_encryption_errors():
        encrypted_doc = self._encryption.encrypt(value=doc, algorithm=algorithm, key_id=key_id, key_alt_name=key_alt_name, query_type=query_type, contention_factor=contention_factor, range_opts=range_opts_bytes, is_expression=is_expression)
        return decode(encrypted_doc)['v']
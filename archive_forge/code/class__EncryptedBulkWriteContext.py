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
class _EncryptedBulkWriteContext(_BulkWriteContext):
    __slots__ = ()

    def __batch_command(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]]) -> tuple[MutableMapping[str, Any], list[Mapping[str, Any]]]:
        namespace = self.db_name + '.$cmd'
        msg, to_send = _encode_batched_write_command(namespace, self.op_type, cmd, docs, self.codec, self)
        if not to_send:
            raise InvalidOperation('cannot do an empty bulk write')
        cmd_start = msg.index(b'\x00', 4) + 9
        outgoing = _inflate_bson(memoryview(msg)[cmd_start:], DEFAULT_RAW_BSON_OPTIONS)
        return (outgoing, to_send)

    def execute(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]], client: MongoClient) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        batched_cmd, to_send = self.__batch_command(cmd, docs)
        result: Mapping[str, Any] = self.conn.command(self.db_name, batched_cmd, codec_options=self.codec, session=self.session, client=client)
        return (result, to_send)

    def execute_unack(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]], client: MongoClient) -> list[Mapping[str, Any]]:
        batched_cmd, to_send = self.__batch_command(cmd, docs)
        self.conn.command(self.db_name, batched_cmd, write_concern=WriteConcern(w=0), session=self.session, client=client)
        return to_send

    @property
    def max_split_size(self) -> int:
        """Reduce the batch splitting size."""
        return _MAX_SPLIT_SIZE_ENC
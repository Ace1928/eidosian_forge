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
class _OpMsg:
    """A MongoDB OP_MSG response message."""
    __slots__ = ('flags', 'cursor_id', 'number_returned', 'payload_document')
    UNPACK_FROM = struct.Struct('<IBi').unpack_from
    OP_CODE = 2013
    CHECKSUM_PRESENT = 1
    MORE_TO_COME = 1 << 1
    EXHAUST_ALLOWED = 1 << 16

    def __init__(self, flags: int, payload_document: bytes):
        self.flags = flags
        self.payload_document = payload_document

    def raw_response(self, cursor_id: Optional[int]=None, user_fields: Optional[Mapping[str, Any]]={}) -> list[Mapping[str, Any]]:
        """
        cursor_id is ignored
        user_fields is used to determine which fields must not be decoded
        """
        inflated_response = _decode_selective(RawBSONDocument(self.payload_document), user_fields, _RAW_ARRAY_BSON_OPTIONS)
        return [inflated_response]

    def unpack_response(self, cursor_id: Optional[int]=None, codec_options: CodecOptions=_UNICODE_REPLACE_CODEC_OPTIONS, user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[dict[str, Any]]:
        """Unpack a OP_MSG command response.

        :Parameters:
          - `cursor_id` (optional): Ignored, for compatibility with _OpReply.
          - `codec_options` (optional): an instance of
            :class:`~bson.codec_options.CodecOptions`
          - `user_fields` (optional): Response fields that should be decoded
            using the TypeDecoders from codec_options, passed to
            bson._decode_all_selective.
        """
        assert not legacy_response
        return bson._decode_all_selective(self.payload_document, codec_options, user_fields)

    def command_response(self, codec_options: CodecOptions) -> dict[str, Any]:
        """Unpack a command response."""
        return self.unpack_response(codec_options=codec_options)[0]

    def raw_command_response(self) -> bytes:
        """Return the bytes of the command response."""
        return self.payload_document

    @property
    def more_to_come(self) -> bool:
        """Is the moreToCome bit set on this response?"""
        return bool(self.flags & self.MORE_TO_COME)

    @classmethod
    def unpack(cls, msg: bytes) -> _OpMsg:
        """Construct an _OpMsg from raw bytes."""
        flags, first_payload_type, first_payload_size = cls.UNPACK_FROM(msg)
        if flags != 0:
            if flags & cls.CHECKSUM_PRESENT:
                raise ProtocolError(f'Unsupported OP_MSG flag checksumPresent: 0x{flags:x}')
            if flags ^ cls.MORE_TO_COME:
                raise ProtocolError(f'Unsupported OP_MSG flags: 0x{flags:x}')
        if first_payload_type != 0:
            raise ProtocolError(f'Unsupported OP_MSG payload type: 0x{first_payload_type:x}')
        if len(msg) != first_payload_size + 5:
            raise ProtocolError('Unsupported OP_MSG reply: >1 section')
        payload_document = msg[5:]
        return cls(flags, payload_document)
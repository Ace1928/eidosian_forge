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
class _OpReply:
    """A MongoDB OP_REPLY response message."""
    __slots__ = ('flags', 'cursor_id', 'number_returned', 'documents')
    UNPACK_FROM = struct.Struct('<iqii').unpack_from
    OP_CODE = 1

    def __init__(self, flags: int, cursor_id: int, number_returned: int, documents: bytes):
        self.flags = flags
        self.cursor_id = Int64(cursor_id)
        self.number_returned = number_returned
        self.documents = documents

    def raw_response(self, cursor_id: Optional[int]=None, user_fields: Optional[Mapping[str, Any]]=None) -> list[bytes]:
        """Check the response header from the database, without decoding BSON.

        Check the response for errors and unpack.

        Can raise CursorNotFound, NotPrimaryError, ExecutionTimeout, or
        OperationFailure.

        :Parameters:
          - `cursor_id` (optional): cursor_id we sent to get this response -
            used for raising an informative exception when we get cursor id not
            valid at server response.
        """
        if self.flags & 1:
            if cursor_id is None:
                raise ProtocolError('No cursor id for getMore operation')
            msg = 'Cursor not found, cursor id: %d' % (cursor_id,)
            errobj = {'ok': 0, 'errmsg': msg, 'code': 43}
            raise CursorNotFound(msg, 43, errobj)
        elif self.flags & 2:
            error_object: dict = bson.BSON(self.documents).decode()
            error_object.setdefault('ok', 0)
            if error_object['$err'].startswith(HelloCompat.LEGACY_ERROR):
                raise NotPrimaryError(error_object['$err'], error_object)
            elif error_object.get('code') == 50:
                default_msg = 'operation exceeded time limit'
                raise ExecutionTimeout(error_object.get('$err', default_msg), error_object.get('code'), error_object)
            raise OperationFailure('database error: %s' % error_object.get('$err'), error_object.get('code'), error_object)
        if self.documents:
            return [self.documents]
        return []

    def unpack_response(self, cursor_id: Optional[int]=None, codec_options: CodecOptions=_UNICODE_REPLACE_CODEC_OPTIONS, user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[dict[str, Any]]:
        """Unpack a response from the database and decode the BSON document(s).

        Check the response for errors and unpack, returning a dictionary
        containing the response data.

        Can raise CursorNotFound, NotPrimaryError, ExecutionTimeout, or
        OperationFailure.

        :Parameters:
          - `cursor_id` (optional): cursor_id we sent to get this response -
            used for raising an informative exception when we get cursor id not
            valid at server response
          - `codec_options` (optional): an instance of
            :class:`~bson.codec_options.CodecOptions`
          - `user_fields` (optional): Response fields that should be decoded
            using the TypeDecoders from codec_options, passed to
            bson._decode_all_selective.
        """
        self.raw_response(cursor_id)
        if legacy_response:
            return bson.decode_all(self.documents, codec_options)
        return bson._decode_all_selective(self.documents, codec_options, user_fields)

    def command_response(self, codec_options: CodecOptions) -> dict[str, Any]:
        """Unpack a command response."""
        docs = self.unpack_response(codec_options=codec_options)
        assert self.number_returned == 1
        return docs[0]

    def raw_command_response(self) -> NoReturn:
        """Return the bytes of the command response."""
        raise NotImplementedError

    @property
    def more_to_come(self) -> bool:
        """Is the moreToCome bit set on this response?"""
        return False

    @classmethod
    def unpack(cls, msg: bytes) -> _OpReply:
        """Construct an _OpReply from raw bytes."""
        flags, cursor_id, _, number_returned = cls.UNPACK_FROM(msg)
        documents = msg[20:]
        return cls(flags, cursor_id, number_returned, documents)
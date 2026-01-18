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
class _BulkWriteContext:
    """A wrapper around Connection for use with write splitting functions."""
    __slots__ = ('db_name', 'conn', 'op_id', 'name', 'field', 'publish', 'start_time', 'listeners', 'session', 'compress', 'op_type', 'codec')

    def __init__(self, database_name: str, cmd_name: str, conn: Connection, operation_id: int, listeners: _EventListeners, session: ClientSession, op_type: int, codec: CodecOptions):
        self.db_name = database_name
        self.conn = conn
        self.op_id = operation_id
        self.listeners = listeners
        self.publish = listeners.enabled_for_commands
        self.name = cmd_name
        self.field = _FIELD_MAP[self.name]
        self.start_time = datetime.datetime.now() if self.publish else None
        self.session = session
        self.compress = bool(conn.compression_context)
        self.op_type = op_type
        self.codec = codec

    def __batch_command(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]]) -> tuple[int, bytes, list[Mapping[str, Any]]]:
        namespace = self.db_name + '.$cmd'
        request_id, msg, to_send = _do_batched_op_msg(namespace, self.op_type, cmd, docs, self.codec, self)
        if not to_send:
            raise InvalidOperation('cannot do an empty bulk write')
        return (request_id, msg, to_send)

    def execute(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]], client: MongoClient) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        request_id, msg, to_send = self.__batch_command(cmd, docs)
        result = self.write_command(cmd, request_id, msg, to_send)
        client._process_response(result, self.session)
        return (result, to_send)

    def execute_unack(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]], client: MongoClient) -> list[Mapping[str, Any]]:
        request_id, msg, to_send = self.__batch_command(cmd, docs)
        self.unack_write(cmd, request_id, msg, 0, to_send)
        return to_send

    @property
    def max_bson_size(self) -> int:
        """A proxy for SockInfo.max_bson_size."""
        return self.conn.max_bson_size

    @property
    def max_message_size(self) -> int:
        """A proxy for SockInfo.max_message_size."""
        if self.compress:
            return self.conn.max_message_size - 16
        return self.conn.max_message_size

    @property
    def max_write_batch_size(self) -> int:
        """A proxy for SockInfo.max_write_batch_size."""
        return self.conn.max_write_batch_size

    @property
    def max_split_size(self) -> int:
        """The maximum size of a BSON command before batch splitting."""
        return self.max_bson_size

    def unack_write(self, cmd: MutableMapping[str, Any], request_id: int, msg: bytes, max_doc_size: int, docs: list[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """A proxy for Connection.unack_write that handles event publishing."""
        if self.publish:
            assert self.start_time is not None
            duration = datetime.datetime.now() - self.start_time
            cmd = self._start(cmd, request_id, docs)
            start = datetime.datetime.now()
        try:
            result = self.conn.unack_write(msg, max_doc_size)
            if self.publish:
                duration = datetime.datetime.now() - start + duration
                if result is not None:
                    reply = _convert_write_result(self.name, cmd, result)
                else:
                    reply = {'ok': 1}
                self._succeed(request_id, reply, duration)
        except Exception as exc:
            if self.publish:
                assert self.start_time is not None
                duration = datetime.datetime.now() - start + duration
                if isinstance(exc, OperationFailure):
                    failure: _DocumentOut = _convert_write_result(self.name, cmd, exc.details)
                elif isinstance(exc, NotPrimaryError):
                    failure = exc.details
                else:
                    failure = _convert_exception(exc)
                self._fail(request_id, failure, duration)
            raise
        finally:
            self.start_time = datetime.datetime.now()
        return result

    @_handle_reauth
    def write_command(self, cmd: MutableMapping[str, Any], request_id: int, msg: bytes, docs: list[Mapping[str, Any]]) -> dict[str, Any]:
        """A proxy for SocketInfo.write_command that handles event publishing."""
        if self.publish:
            assert self.start_time is not None
            duration = datetime.datetime.now() - self.start_time
            self._start(cmd, request_id, docs)
            start = datetime.datetime.now()
        try:
            reply = self.conn.write_command(request_id, msg, self.codec)
            if self.publish:
                duration = datetime.datetime.now() - start + duration
                self._succeed(request_id, reply, duration)
        except Exception as exc:
            if self.publish:
                duration = datetime.datetime.now() - start + duration
                if isinstance(exc, (NotPrimaryError, OperationFailure)):
                    failure: _DocumentOut = exc.details
                else:
                    failure = _convert_exception(exc)
                self._fail(request_id, failure, duration)
            raise
        finally:
            self.start_time = datetime.datetime.now()
        return reply

    def _start(self, cmd: MutableMapping[str, Any], request_id: int, docs: list[Mapping[str, Any]]) -> MutableMapping[str, Any]:
        """Publish a CommandStartedEvent."""
        cmd[self.field] = docs
        self.listeners.publish_command_start(cmd, self.db_name, request_id, self.conn.address, self.op_id, self.conn.service_id)
        return cmd

    def _succeed(self, request_id: int, reply: _DocumentOut, duration: timedelta) -> None:
        """Publish a CommandSucceededEvent."""
        self.listeners.publish_command_success(duration, reply, self.name, request_id, self.conn.address, self.op_id, self.conn.service_id, database_name=self.db_name)

    def _fail(self, request_id: int, failure: _DocumentOut, duration: timedelta) -> None:
        """Publish a CommandFailedEvent."""
        self.listeners.publish_command_failure(duration, failure, self.name, request_id, self.conn.address, self.op_id, self.conn.service_id, database_name=self.db_name)
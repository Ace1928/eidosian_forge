from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
class _ReadStream(io.RawIOBase):

    def __init__(self, int_buffer: JsArray, byte_buffer: JsArray, timeout: float, worker: JsProxy, connection_id: int, request: EmscriptenRequest):
        self.int_buffer = int_buffer
        self.byte_buffer = byte_buffer
        self.read_pos = 0
        self.read_len = 0
        self.connection_id = connection_id
        self.worker = worker
        self.timeout = int(1000 * timeout) if timeout > 0 else None
        self.is_live = True
        self._is_closed = False
        self.request: EmscriptenRequest | None = request

    def __del__(self) -> None:
        self.close()

    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def closed(self) -> bool:
        return self.is_closed()

    def close(self) -> None:
        if not self.is_closed():
            self.read_len = 0
            self.read_pos = 0
            self.int_buffer = None
            self.byte_buffer = None
            self._is_closed = True
            self.request = None
            if self.is_live:
                self.worker.postMessage(_obj_from_dict({'close': self.connection_id}))
                self.is_live = False
            super().close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def readinto(self, byte_obj: Buffer) -> int:
        if not self.int_buffer:
            raise _StreamingError('No buffer for stream in _ReadStream.readinto', request=self.request, response=None)
        if self.read_len == 0:
            js.Atomics.store(self.int_buffer, 0, ERROR_TIMEOUT)
            self.worker.postMessage(_obj_from_dict({'getMore': self.connection_id}))
            if js.Atomics.wait(self.int_buffer, 0, ERROR_TIMEOUT, self.timeout) == 'timed-out':
                raise _TimeoutError
            data_len = self.int_buffer[0]
            if data_len > 0:
                self.read_len = data_len
                self.read_pos = 0
            elif data_len == ERROR_EXCEPTION:
                string_len = self.int_buffer[1]
                js_decoder = js.TextDecoder.new()
                json_str = js_decoder.decode(self.byte_buffer.slice(0, string_len))
                raise _StreamingError(f'Exception thrown in fetch: {json_str}', request=self.request, response=None)
            else:
                self.is_live = False
                self.close()
                return 0
        ret_length = min(self.read_len, len(memoryview(byte_obj)))
        subarray = self.byte_buffer.subarray(self.read_pos, self.read_pos + ret_length).to_py()
        memoryview(byte_obj)[0:ret_length] = subarray
        self.read_len -= ret_length
        self.read_pos += ret_length
        return ret_length
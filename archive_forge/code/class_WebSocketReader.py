import asyncio
import functools
import json
import random
import re
import sys
import zlib
from enum import IntEnum
from struct import Struct
from typing import (
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .helpers import NO_EXTENSIONS
from .streams import DataQueue
class WebSocketReader:

    def __init__(self, queue: DataQueue[WSMessage], max_msg_size: int, compress: bool=True) -> None:
        self.queue = queue
        self._max_msg_size = max_msg_size
        self._exc: Optional[BaseException] = None
        self._partial = bytearray()
        self._state = WSParserState.READ_HEADER
        self._opcode: Optional[int] = None
        self._frame_fin = False
        self._frame_opcode: Optional[int] = None
        self._frame_payload = bytearray()
        self._tail = b''
        self._has_mask = False
        self._frame_mask: Optional[bytes] = None
        self._payload_length = 0
        self._payload_length_flag = 0
        self._compressed: Optional[bool] = None
        self._decompressobj: Optional[ZLibDecompressor] = None
        self._compress = compress

    def feed_eof(self) -> None:
        self.queue.feed_eof()

    def feed_data(self, data: bytes) -> Tuple[bool, bytes]:
        if self._exc:
            return (True, data)
        try:
            return self._feed_data(data)
        except Exception as exc:
            self._exc = exc
            self.queue.set_exception(exc)
            return (True, b'')

    def _feed_data(self, data: bytes) -> Tuple[bool, bytes]:
        for fin, opcode, payload, compressed in self.parse_frame(data):
            if compressed and (not self._decompressobj):
                self._decompressobj = ZLibDecompressor(suppress_deflate_header=True)
            if opcode == WSMsgType.CLOSE:
                if len(payload) >= 2:
                    close_code = UNPACK_CLOSE_CODE(payload[:2])[0]
                    if close_code < 3000 and close_code not in ALLOWED_CLOSE_CODES:
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, f'Invalid close code: {close_code}')
                    try:
                        close_message = payload[2:].decode('utf-8')
                    except UnicodeDecodeError as exc:
                        raise WebSocketError(WSCloseCode.INVALID_TEXT, 'Invalid UTF-8 text message') from exc
                    msg = WSMessage(WSMsgType.CLOSE, close_code, close_message)
                elif payload:
                    raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, f'Invalid close frame: {fin} {opcode} {payload!r}')
                else:
                    msg = WSMessage(WSMsgType.CLOSE, 0, '')
                self.queue.feed_data(msg, 0)
            elif opcode == WSMsgType.PING:
                self.queue.feed_data(WSMessage(WSMsgType.PING, payload, ''), len(payload))
            elif opcode == WSMsgType.PONG:
                self.queue.feed_data(WSMessage(WSMsgType.PONG, payload, ''), len(payload))
            elif opcode not in (WSMsgType.TEXT, WSMsgType.BINARY) and self._opcode is None:
                raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, f'Unexpected opcode={opcode!r}')
            elif not fin:
                if opcode != WSMsgType.CONTINUATION:
                    self._opcode = opcode
                self._partial.extend(payload)
                if self._max_msg_size and len(self._partial) >= self._max_msg_size:
                    raise WebSocketError(WSCloseCode.MESSAGE_TOO_BIG, 'Message size {} exceeds limit {}'.format(len(self._partial), self._max_msg_size))
            else:
                if self._partial:
                    if opcode != WSMsgType.CONTINUATION:
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'The opcode in non-fin frame is expected to be zero, got {!r}'.format(opcode))
                if opcode == WSMsgType.CONTINUATION:
                    assert self._opcode is not None
                    opcode = self._opcode
                    self._opcode = None
                self._partial.extend(payload)
                if self._max_msg_size and len(self._partial) >= self._max_msg_size:
                    raise WebSocketError(WSCloseCode.MESSAGE_TOO_BIG, 'Message size {} exceeds limit {}'.format(len(self._partial), self._max_msg_size))
                if compressed:
                    assert self._decompressobj is not None
                    self._partial.extend(_WS_DEFLATE_TRAILING)
                    payload_merged = self._decompressobj.decompress_sync(self._partial, self._max_msg_size)
                    if self._decompressobj.unconsumed_tail:
                        left = len(self._decompressobj.unconsumed_tail)
                        raise WebSocketError(WSCloseCode.MESSAGE_TOO_BIG, 'Decompressed message size {} exceeds limit {}'.format(self._max_msg_size + left, self._max_msg_size))
                else:
                    payload_merged = bytes(self._partial)
                self._partial.clear()
                if opcode == WSMsgType.TEXT:
                    try:
                        text = payload_merged.decode('utf-8')
                        self.queue.feed_data(WSMessage(WSMsgType.TEXT, text, ''), len(text))
                    except UnicodeDecodeError as exc:
                        raise WebSocketError(WSCloseCode.INVALID_TEXT, 'Invalid UTF-8 text message') from exc
                else:
                    self.queue.feed_data(WSMessage(WSMsgType.BINARY, payload_merged, ''), len(payload_merged))
        return (False, b'')

    def parse_frame(self, buf: bytes) -> List[Tuple[bool, Optional[int], bytearray, Optional[bool]]]:
        """Return the next frame from the socket."""
        frames = []
        if self._tail:
            buf, self._tail = (self._tail + buf, b'')
        start_pos = 0
        buf_length = len(buf)
        while True:
            if self._state == WSParserState.READ_HEADER:
                if buf_length - start_pos >= 2:
                    data = buf[start_pos:start_pos + 2]
                    start_pos += 2
                    first_byte, second_byte = data
                    fin = first_byte >> 7 & 1
                    rsv1 = first_byte >> 6 & 1
                    rsv2 = first_byte >> 5 & 1
                    rsv3 = first_byte >> 4 & 1
                    opcode = first_byte & 15
                    if rsv2 or rsv3 or (rsv1 and (not self._compress)):
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received frame with non-zero reserved bits')
                    if opcode > 7 and fin == 0:
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received fragmented control frame')
                    has_mask = second_byte >> 7 & 1
                    length = second_byte & 127
                    if opcode > 7 and length > 125:
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Control frame payload cannot be larger than 125 bytes')
                    if self._frame_fin or self._compressed is None:
                        self._compressed = True if rsv1 else False
                    elif rsv1:
                        raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received frame with non-zero reserved bits')
                    self._frame_fin = bool(fin)
                    self._frame_opcode = opcode
                    self._has_mask = bool(has_mask)
                    self._payload_length_flag = length
                    self._state = WSParserState.READ_PAYLOAD_LENGTH
                else:
                    break
            if self._state == WSParserState.READ_PAYLOAD_LENGTH:
                length = self._payload_length_flag
                if length == 126:
                    if buf_length - start_pos >= 2:
                        data = buf[start_pos:start_pos + 2]
                        start_pos += 2
                        length = UNPACK_LEN2(data)[0]
                        self._payload_length = length
                        self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
                    else:
                        break
                elif length > 126:
                    if buf_length - start_pos >= 8:
                        data = buf[start_pos:start_pos + 8]
                        start_pos += 8
                        length = UNPACK_LEN3(data)[0]
                        self._payload_length = length
                        self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
                    else:
                        break
                else:
                    self._payload_length = length
                    self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
            if self._state == WSParserState.READ_PAYLOAD_MASK:
                if buf_length - start_pos >= 4:
                    self._frame_mask = buf[start_pos:start_pos + 4]
                    start_pos += 4
                    self._state = WSParserState.READ_PAYLOAD
                else:
                    break
            if self._state == WSParserState.READ_PAYLOAD:
                length = self._payload_length
                payload = self._frame_payload
                chunk_len = buf_length - start_pos
                if length >= chunk_len:
                    self._payload_length = length - chunk_len
                    payload.extend(buf[start_pos:])
                    start_pos = buf_length
                else:
                    self._payload_length = 0
                    payload.extend(buf[start_pos:start_pos + length])
                    start_pos = start_pos + length
                if self._payload_length == 0:
                    if self._has_mask:
                        assert self._frame_mask is not None
                        _websocket_mask(self._frame_mask, payload)
                    frames.append((self._frame_fin, self._frame_opcode, payload, self._compressed))
                    self._frame_payload = bytearray()
                    self._state = WSParserState.READ_HEADER
                else:
                    break
        self._tail = buf[start_pos:]
        return frames
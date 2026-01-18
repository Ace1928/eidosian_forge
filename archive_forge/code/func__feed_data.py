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
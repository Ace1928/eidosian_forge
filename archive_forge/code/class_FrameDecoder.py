import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
class FrameDecoder:

    def __init__(self, client: bool, extensions: Optional[List['Extension']]=None) -> None:
        self.client = client
        self.extensions = extensions or []
        self.buffer = Buffer()
        self.header: Optional[Header] = None
        self.effective_opcode: Optional[Opcode] = None
        self.masker: Union[None, XorMaskerNull, XorMaskerSimple] = None
        self.payload_required = 0
        self.payload_consumed = 0

    def receive_bytes(self, data: bytes) -> None:
        self.buffer.feed(data)

    def process_buffer(self) -> Optional[Frame]:
        if not self.header:
            if not self.parse_header():
                return None
        assert self.header is not None
        assert self.masker is not None
        assert self.effective_opcode is not None
        if len(self.buffer) < self.payload_required:
            return None
        payload_remaining = self.header.payload_len - self.payload_consumed
        payload = self.buffer.consume_at_most(payload_remaining)
        if not payload and self.header.payload_len > 0:
            return None
        self.buffer.commit()
        self.payload_consumed += len(payload)
        finished = self.payload_consumed == self.header.payload_len
        payload = self.masker.process(payload)
        for extension in self.extensions:
            payload_ = extension.frame_inbound_payload_data(self, payload)
            if isinstance(payload_, CloseReason):
                raise ParseFailed('error in extension', payload_)
            payload = payload_
        if finished:
            final = bytearray()
            for extension in self.extensions:
                result = extension.frame_inbound_complete(self, self.header.fin)
                if isinstance(result, CloseReason):
                    raise ParseFailed('error in extension', result)
                if result is not None:
                    final += result
            payload += final
        frame = Frame(self.effective_opcode, payload, finished, self.header.fin)
        if finished:
            self.header = None
            self.effective_opcode = None
            self.masker = None
        else:
            self.effective_opcode = Opcode.CONTINUATION
        return frame

    def parse_header(self) -> bool:
        data = self.buffer.consume_exactly(2)
        if data is None:
            self.buffer.rollback()
            return False
        fin = bool(data[0] & FIN_MASK)
        rsv = RsvBits(bool(data[0] & RSV1_MASK), bool(data[0] & RSV2_MASK), bool(data[0] & RSV3_MASK))
        opcode = data[0] & OPCODE_MASK
        try:
            opcode = Opcode(opcode)
        except ValueError:
            raise ParseFailed(f'Invalid opcode {opcode:#x}')
        if opcode.iscontrol() and (not fin):
            raise ParseFailed('Invalid attempt to fragment control frame')
        has_mask = bool(data[1] & MASK_MASK)
        payload_len_short = data[1] & PAYLOAD_LEN_MASK
        payload_len = self.parse_extended_payload_length(opcode, payload_len_short)
        if payload_len is None:
            self.buffer.rollback()
            return False
        self.extension_processing(opcode, rsv, payload_len)
        if has_mask and self.client:
            raise ParseFailed('client received unexpected masked frame')
        if not has_mask and (not self.client):
            raise ParseFailed('server received unexpected unmasked frame')
        if has_mask:
            masking_key = self.buffer.consume_exactly(4)
            if masking_key is None:
                self.buffer.rollback()
                return False
            self.masker = XorMaskerSimple(masking_key)
        else:
            self.masker = XorMaskerNull()
        self.buffer.commit()
        self.header = Header(fin, rsv, opcode, payload_len, None)
        self.effective_opcode = self.header.opcode
        if self.header.opcode.iscontrol():
            self.payload_required = payload_len
        else:
            self.payload_required = 0
        self.payload_consumed = 0
        return True

    def parse_extended_payload_length(self, opcode: Opcode, payload_len: int) -> Optional[int]:
        if opcode.iscontrol() and payload_len > MAX_PAYLOAD_NORMAL:
            raise ParseFailed('Control frame with payload len > 125')
        if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
            data = self.buffer.consume_exactly(2)
            if data is None:
                return None
            payload_len, = struct.unpack('!H', data)
            if payload_len <= MAX_PAYLOAD_NORMAL:
                raise ParseFailed('Payload length used 2 bytes when 1 would have sufficed')
        elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
            data = self.buffer.consume_exactly(8)
            if data is None:
                return None
            payload_len, = struct.unpack('!Q', data)
            if payload_len <= MAX_PAYLOAD_TWO_BYTE:
                raise ParseFailed('Payload length used 8 bytes when 2 would have sufficed')
            if payload_len >> 63:
                raise ParseFailed('8-byte payload length with non-zero MSB')
        return payload_len

    def extension_processing(self, opcode: Opcode, rsv: RsvBits, payload_len: int) -> None:
        rsv_used = [False, False, False]
        for extension in self.extensions:
            result = extension.frame_inbound_header(self, opcode, rsv, payload_len)
            if isinstance(result, CloseReason):
                raise ParseFailed('error in extension', result)
            for bit, used in enumerate(result):
                if used:
                    rsv_used[bit] = True
        for expected, found in zip(rsv_used, rsv):
            if found and (not expected):
                raise ParseFailed('Reserved bit set unexpectedly')
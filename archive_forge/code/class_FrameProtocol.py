import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
class FrameProtocol:

    def __init__(self, client: bool, extensions: List['Extension']) -> None:
        self.client = client
        self.extensions = [ext for ext in extensions if ext.enabled()]
        self._frame_decoder = FrameDecoder(self.client, self.extensions)
        self._message_decoder = MessageDecoder()
        self._parse_more = self._parse_more_gen()
        self._outbound_opcode: Optional[Opcode] = None

    def _process_close(self, frame: Frame) -> Frame:
        data = frame.payload
        assert isinstance(data, (bytes, bytearray))
        if not data:
            data = (CloseReason.NO_STATUS_RCVD, '')
        elif len(data) == 1:
            raise ParseFailed('CLOSE with 1 byte payload')
        else:
            code, = struct.unpack('!H', data[:2])
            if code < MIN_CLOSE_REASON or code > MAX_CLOSE_REASON:
                raise ParseFailed('CLOSE with invalid code')
            try:
                code = CloseReason(code)
            except ValueError:
                pass
            if code in LOCAL_ONLY_CLOSE_REASONS:
                raise ParseFailed('remote CLOSE with local-only reason')
            if not isinstance(code, CloseReason) and code <= MAX_PROTOCOL_CLOSE_REASON:
                raise ParseFailed('CLOSE with unknown reserved code')
            try:
                reason = data[2:].decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ParseFailed('Error decoding CLOSE reason: ' + str(exc), CloseReason.INVALID_FRAME_PAYLOAD_DATA)
            data = (code, reason)
        return Frame(frame.opcode, data, frame.frame_finished, frame.message_finished)

    def _parse_more_gen(self) -> Generator[Optional[Frame], None, None]:
        self.extensions = [ext for ext in self.extensions if ext.enabled()]
        closed = False
        while not closed:
            frame = self._frame_decoder.process_buffer()
            if frame is not None:
                if not frame.opcode.iscontrol():
                    frame = self._message_decoder.process_frame(frame)
                elif frame.opcode == Opcode.CLOSE:
                    frame = self._process_close(frame)
                    closed = True
            yield frame

    def receive_bytes(self, data: bytes) -> None:
        self._frame_decoder.receive_bytes(data)

    def received_frames(self) -> Generator[Frame, None, None]:
        for event in self._parse_more:
            if event is None:
                break
            else:
                yield event

    def close(self, code: Optional[int]=None, reason: Optional[str]=None) -> bytes:
        payload = bytearray()
        if code is CloseReason.NO_STATUS_RCVD:
            code = None
        if code is None and reason:
            raise TypeError('cannot specify a reason without a code')
        if code in LOCAL_ONLY_CLOSE_REASONS:
            code = CloseReason.NORMAL_CLOSURE
        if code is not None:
            payload += bytearray(struct.pack('!H', code))
            if reason is not None:
                payload += _truncate_utf8(reason.encode('utf-8'), MAX_PAYLOAD_NORMAL - 2)
        return self._serialize_frame(Opcode.CLOSE, payload)

    def ping(self, payload: bytes=b'') -> bytes:
        return self._serialize_frame(Opcode.PING, payload)

    def pong(self, payload: bytes=b'') -> bytes:
        return self._serialize_frame(Opcode.PONG, payload)

    def send_data(self, payload: Union[bytes, bytearray, str]=b'', fin: bool=True) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            opcode = Opcode.BINARY
        elif isinstance(payload, str):
            opcode = Opcode.TEXT
            payload = payload.encode('utf-8')
        else:
            raise ValueError('Must provide bytes or text')
        if self._outbound_opcode is None:
            self._outbound_opcode = opcode
        elif self._outbound_opcode is not opcode:
            raise TypeError('Data type mismatch inside message')
        else:
            opcode = Opcode.CONTINUATION
        if fin:
            self._outbound_opcode = None
        return self._serialize_frame(opcode, payload, fin)

    def _make_fin_rsv_opcode(self, fin: bool, rsv: RsvBits, opcode: Opcode) -> int:
        fin_bits = int(fin) << 7
        rsv_bits = (int(rsv.rsv1) << 6) + (int(rsv.rsv2) << 5) + (int(rsv.rsv3) << 4)
        opcode_bits = int(opcode)
        return fin_bits | rsv_bits | opcode_bits

    def _serialize_frame(self, opcode: Opcode, payload: bytes=b'', fin: bool=True) -> bytes:
        rsv = RsvBits(False, False, False)
        for extension in reversed(self.extensions):
            rsv, payload = extension.frame_outbound(self, opcode, rsv, payload, fin)
        fin_rsv_opcode = self._make_fin_rsv_opcode(fin, rsv, opcode)
        payload_length = len(payload)
        quad_payload = False
        if payload_length <= MAX_PAYLOAD_NORMAL:
            first_payload = payload_length
            second_payload = None
        elif payload_length <= MAX_PAYLOAD_TWO_BYTE:
            first_payload = PAYLOAD_LENGTH_TWO_BYTE
            second_payload = payload_length
        else:
            first_payload = PAYLOAD_LENGTH_EIGHT_BYTE
            second_payload = payload_length
            quad_payload = True
        if self.client:
            first_payload |= 1 << 7
        header = bytearray([fin_rsv_opcode, first_payload])
        if second_payload is not None:
            if opcode.iscontrol():
                raise ValueError('payload too long for control frame')
            if quad_payload:
                header += bytearray(struct.pack('!Q', second_payload))
            else:
                header += bytearray(struct.pack('!H', second_payload))
        if self.client:
            masking_key = os.urandom(4)
            masker = XorMaskerSimple(masking_key)
            return header + masking_key + masker.process(payload)
        return header + payload
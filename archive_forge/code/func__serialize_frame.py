import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
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
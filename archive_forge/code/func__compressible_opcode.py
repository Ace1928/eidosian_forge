import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def _compressible_opcode(self, opcode: Opcode) -> bool:
    return opcode in (Opcode.TEXT, Opcode.BINARY, Opcode.CONTINUATION)
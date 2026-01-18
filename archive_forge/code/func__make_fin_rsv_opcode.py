import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def _make_fin_rsv_opcode(self, fin: bool, rsv: RsvBits, opcode: Opcode) -> int:
    fin_bits = int(fin) << 7
    rsv_bits = (int(rsv.rsv1) << 6) + (int(rsv.rsv2) << 5) + (int(rsv.rsv3) << 4)
    opcode_bits = int(opcode)
    return fin_bits | rsv_bits | opcode_bits
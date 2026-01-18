import struct
from typing import Union
class ARMTDecoder(BCJFilter):

    def __init__(self, size: int):
        super().__init__(self.armt_code, 4, False, size)
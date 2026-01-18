import struct
from typing import Union
class PPCDecoder(BCJFilter):

    def __init__(self, size: int):
        super().__init__(self.ppc_code, 4, False, size)
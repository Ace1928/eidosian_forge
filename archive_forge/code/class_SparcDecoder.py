import struct
from typing import Union
class SparcDecoder(BCJFilter):

    def __init__(self, size: int):
        super().__init__(self.sparc_code, 4, False, size)
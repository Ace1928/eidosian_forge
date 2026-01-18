import struct
from typing import Union
class BCJDecoder(BCJFilter):

    def __init__(self, size: int):
        super().__init__(self.x86_code, 5, False, size)
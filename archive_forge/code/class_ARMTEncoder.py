import struct
from typing import Union
class ARMTEncoder(BCJFilter):

    def __init__(self):
        super().__init__(self.armt_code, 4, True)
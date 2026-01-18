import struct
from pyu2f import errors
def InternalEncodeLc(self):
    dl = 0
    if self.data:
        dl = len(self.data)
    fourbyte = struct.pack('>I', dl)
    return bytearray(fourbyte[1:])
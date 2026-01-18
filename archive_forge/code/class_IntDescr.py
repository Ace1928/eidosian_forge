import base64
import struct
from os_ken.lib import addrconv
class IntDescr(TypeDescr):

    def __init__(self, size):
        self.size = size

    def to_user(self, binary):
        i = 0
        for _ in range(self.size):
            c = binary[:1]
            i = i * 256 + ord(c)
            binary = binary[1:]
        return i

    def from_user(self, i):
        binary = b''
        for _ in range(self.size):
            binary = struct.Struct('>B').pack(i & 255) + binary
            i //= 256
        return binary
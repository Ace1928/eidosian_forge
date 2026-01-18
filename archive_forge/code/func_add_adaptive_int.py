import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_adaptive_int(self, n):
    """
        Add an integer to the stream.

        :param int n: integer to add
        """
    if n >= Message.big_int:
        self.packet.write(max_byte)
        self.add_string(util.deflate_long(n))
    else:
        self.packet.write(struct.pack('>I', n))
    return self
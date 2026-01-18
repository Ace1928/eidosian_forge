import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_byte(self, b):
    """
        Write a single byte to the stream, without any formatting.

        :param bytes b: byte to add
        """
    self.packet.write(b)
    return self
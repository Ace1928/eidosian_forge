import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_bytes(self, b):
    """
        Write bytes to the stream, without any formatting.

        :param bytes b: bytes to add
        """
    self.packet.write(b)
    return self
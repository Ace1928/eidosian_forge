import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_boolean(self, b):
    """
        Add a boolean value to the stream.

        :param bool b: boolean value to add
        """
    if b:
        self.packet.write(one_byte)
    else:
        self.packet.write(zero_byte)
    return self
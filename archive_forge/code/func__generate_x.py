import os
from hashlib import sha1
from paramiko import util
from paramiko.common import max_byte, zero_byte, byte_chr, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _generate_x(self):
    while 1:
        x_bytes = os.urandom(128)
        x_bytes = byte_mask(x_bytes[0], 127) + x_bytes[1:]
        if x_bytes[:8] != b7fffffffffffffff and x_bytes[:8] != b0000000000000000:
            break
    self.x = util.inflate_long(x_bytes)
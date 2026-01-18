import select
import socket
import struct
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord
from paramiko.message import Message
def _send_version(self):
    m = Message()
    m.add_int(_VERSION)
    self._send_packet(CMD_INIT, m)
    t, data = self._read_packet()
    if t != CMD_VERSION:
        raise SFTPError('Incompatible sftp protocol')
    version = struct.unpack('>I', data[:4])[0]
    return version
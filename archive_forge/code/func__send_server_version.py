import select
import socket
import struct
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord
from paramiko.message import Message
def _send_server_version(self):
    t, data = self._read_packet()
    if t != CMD_INIT:
        raise SFTPError('Incompatible sftp protocol')
    version = struct.unpack('>I', data[:4])[0]
    extension_pairs = ['check-file', 'md5,sha1']
    msg = Message()
    msg.add_int(_VERSION)
    msg.add(*extension_pairs)
    self._send_packet(CMD_VERSION, msg)
    return version
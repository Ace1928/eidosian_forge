import os
import errno
import sys
from hashlib import md5, sha1
from paramiko import util
from paramiko.sftp import (
from paramiko.sftp_si import SFTPServerInterface
from paramiko.sftp_attr import SFTPAttributes
from paramiko.common import DEBUG
from paramiko.server import SubsystemHandler
from paramiko.util import b
from paramiko.sftp import (
from paramiko.sftp_handle import SFTPHandle
def _read_folder(self, request_number, folder):
    flist = folder._get_next_files()
    if len(flist) == 0:
        self._send_status(request_number, SFTP_EOF)
        return
    msg = Message()
    msg.add_int(request_number)
    msg.add_int(len(flist))
    for attr in flist:
        msg.add_string(attr.filename)
        msg.add_string(attr)
        attr._pack(msg)
    self._send_packet(CMD_NAME, msg)
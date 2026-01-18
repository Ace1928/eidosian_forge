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
def _send_handle_response(self, request_number, handle, folder=False):
    if not issubclass(type(handle), SFTPHandle):
        self._send_status(request_number, handle)
        return
    handle._set_name(b('hx{:d}'.format(self.next_handle)))
    self.next_handle += 1
    if folder:
        self.folder_table[handle._get_name()] = handle
    else:
        self.file_table[handle._get_name()] = handle
    self._response(request_number, CMD_HANDLE, handle._get_name())
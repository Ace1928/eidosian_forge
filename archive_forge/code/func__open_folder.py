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
def _open_folder(self, request_number, path):
    resp = self.server.list_folder(path)
    if issubclass(type(resp), list):
        folder = SFTPHandle()
        folder._set_files(resp)
        self._send_handle_response(request_number, folder, True)
        return
    self._send_status(request_number, resp)
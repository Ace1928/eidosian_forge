import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SFTPServerWithoutSSH(SFTPServer):
    """An SFTP server that uses a simple TCP socket pair rather than SSH."""

    def __init__(self):
        super().__init__()
        self._vendor = ssh.LoopbackVendor()
        self.request_handler_class = TestingSFTPWithoutSSHConnectionHandler

    def get_host_key(self):
        return None
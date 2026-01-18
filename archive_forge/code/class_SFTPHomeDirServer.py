import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SFTPHomeDirServer(SFTPServerWithoutSSH):
    """A test server for sftp transports, using homedir relative urls."""

    def get_url(self):
        """See breezy.transport.Server.get_url."""
        return self._get_sftp_url('%7E/')
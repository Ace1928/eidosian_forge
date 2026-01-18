import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class StubSFTPHandle(paramiko.SFTPHandle):

    def stat(self):
        try:
            return paramiko.SFTPAttributes.from_stat(os.fstat(self.readfile.fileno()))
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def chattr(self, attr):
        trace.mutter('Changing permissions on %s to %s', self.filename, attr)
        try:
            paramiko.SFTPServer.set_file_attr(self.filename, attr)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)
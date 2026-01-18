import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SFTPServer(test_server.TestingTCPServerInAThread):
    """Common code for SFTP server facilities."""

    def __init__(self, server_interface=StubServer):
        self.host = '127.0.0.1'
        self.port = 0
        super().__init__((self.host, self.port), TestingSFTPServer, TestingSFTPConnectionHandler)
        self._original_vendor = None
        self._vendor = ssh.ParamikoVendor()
        self._server_interface = server_interface
        self._host_key = None
        self.logs = []
        self.add_latency = 0
        self._homedir = None
        self._server_homedir = None
        self._root = None

    def _get_sftp_url(self, path):
        """Calculate an sftp url to this server for path."""
        return 'sftp://foo:bar@{}:{}/{}'.format(self.host, self.port, path)

    def log(self, message):
        """StubServer uses this to log when a new server is created."""
        self.logs.append(message)

    def create_server(self):
        server = self.server_class((self.host, self.port), self.request_handler_class, self)
        return server

    def get_host_key(self):
        if self._host_key is None:
            key_file = osutils.pathjoin(self._homedir, 'test_rsa.key')
            f = open(key_file, 'w')
            try:
                f.write(STUB_SERVER_KEY)
            finally:
                f.close()
            self._host_key = paramiko.RSAKey.from_private_key_file(key_file)
        return self._host_key

    def start_server(self, backing_server=None):
        if not (backing_server is None or isinstance(backing_server, test_server.LocalURLServer)):
            raise AssertionError('backing_server should not be %r, because this can only serve the local current working directory.' % (backing_server,))
        self._original_vendor = ssh._ssh_vendor_manager._cached_ssh_vendor
        ssh._ssh_vendor_manager._cached_ssh_vendor = self._vendor
        self._homedir = osutils.getcwd()
        if sys.platform == 'win32':
            self._homedir = osutils.normpath(self._homedir)
        else:
            self._homedir = self._homedir
        if self._server_homedir is None:
            self._server_homedir = self._homedir
        self._root = '/'
        if sys.platform == 'win32':
            self._root = ''
        super().start_server()

    def stop_server(self):
        try:
            super().stop_server()
        finally:
            ssh._ssh_vendor_manager._cached_ssh_vendor = self._original_vendor

    def get_bogus_url(self):
        """See breezy.transport.Server.get_bogus_url."""
        s = socket.socket()
        s.bind(('localhost', 0))
        return 'sftp://%s:%s/' % s.getsockname()
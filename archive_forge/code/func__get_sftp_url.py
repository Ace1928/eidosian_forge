import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def _get_sftp_url(self, path):
    """Calculate an sftp url to this server for path."""
    return 'sftp://foo:bar@{}:{}/{}'.format(self.host, self.port, path)
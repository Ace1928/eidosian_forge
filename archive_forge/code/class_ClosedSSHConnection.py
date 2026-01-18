import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class ClosedSSHConnection(ssh.SSHConnection):
    """An SSH connection that just has closed channels."""

    def __init__(self, vendor):
        self.vendor = vendor

    def close(self):
        self.vendor.calls.append(('close',))

    def get_sock_or_pipes(self):
        bzr_read, ssh_write = create_file_pipes()
        ssh_write.close()
        if self.vendor.fail_at_write:
            ssh_read, bzr_write = create_file_pipes()
            ssh_read.close()
        else:
            bzr_write = self.vendor.write_to
        return ('pipes', (bzr_read, bzr_write))
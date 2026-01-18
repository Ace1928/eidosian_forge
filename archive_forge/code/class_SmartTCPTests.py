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
class SmartTCPTests(tests.TestCase):
    """Tests for connection/end to end behaviour using the TCP server.

    All of these tests are run with a server running in another thread serving
    a MemoryTransport, and a connection to it already open.

    the server is obtained by calling self.start_server(readonly=False).
    """

    def start_server(self, readonly=False, backing_transport=None):
        """Setup the server.

        :param readonly: Create a readonly server.
        """
        if backing_transport is None:
            mem_server = memory.MemoryServer()
            mem_server.start_server()
            self.addCleanup(mem_server.stop_server)
            self.permit_url(mem_server.get_url())
            self.backing_transport = _mod_transport.get_transport_from_url(mem_server.get_url())
        else:
            self.backing_transport = backing_transport
        if readonly:
            self.real_backing_transport = self.backing_transport
            self.backing_transport = _mod_transport.get_transport_from_url('readonly+' + self.backing_transport.abspath('.'))
        self.server = _mod_server.SmartTCPServer(self.backing_transport, client_timeout=4.0)
        self.server.start_server('127.0.0.1', 0)
        self.server.start_background_thread('-' + self.id())
        self.transport = remote.RemoteTCPTransport(self.server.get_url())
        self.addCleanup(self.stop_server)
        self.permit_url(self.server.get_url())

    def stop_server(self):
        """Disconnect the client and stop the server.

        This must be re-entrant as some tests will call it explicitly in
        addition to the normal cleanup.
        """
        if getattr(self, 'transport', None):
            self.transport.disconnect()
            del self.transport
        if getattr(self, 'server', None):
            self.server.stop_background_thread()
            del self.server
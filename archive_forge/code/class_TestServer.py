import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestServer(transport.Server):
    """A Transport Server dedicated to tests.

    The TestServer interface provides a server for a given transport. We use
    these servers as loopback testing tools. For any given transport the
    Servers it provides must either allow writing, or serve the contents
    of osutils.getcwd() at the time start_server is called.

    Note that these are real servers - they must implement all the things
    that we want bzr transports to take advantage of.
    """

    def get_url(self):
        """Return a url for this server.

        If the transport does not represent a disk directory (i.e. it is
        a database like svn, or a memory only transport, it should return
        a connection to a newly established resource for this Server.
        Otherwise it should return a url that will provide access to the path
        that was osutils.getcwd() when start_server() was called.

        Subsequent calls will return the same resource.
        """
        raise NotImplementedError

    def get_bogus_url(self):
        """Return a url for this protocol, that will fail to connect.

        This may raise NotImplementedError to indicate that this server cannot
        provide bogus urls.
        """
        raise NotImplementedError
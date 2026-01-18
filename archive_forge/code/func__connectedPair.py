import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def _connectedPair(self):
    """
        Return the two sockets which make up a new TCP connection.
        """
    client, server = socketpair()
    self.addCleanup(client.close)
    self.addCleanup(server.close)
    return (client, server)
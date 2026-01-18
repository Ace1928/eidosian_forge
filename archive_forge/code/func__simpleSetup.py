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
def _simpleSetup(self):
    reactor = self.buildReactor()
    client, server = self._connectedPair()
    fd = FileDescriptor(reactor)
    fd.fileno = client.fileno
    return (reactor, fd, server)
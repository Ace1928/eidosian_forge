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
class Victim(FileDescriptor):
    """
            This L{FileDescriptor} will have its socket closed out from under it
            and another socket will take its place.  It will raise a
            socket.error from C{fileno} after this happens (because socket
            objects remember whether they have been closed), so as long as the
            reactor calls the C{fileno} method the problem will be detected.
            """

    def fileno(self):
        return server.fileno()

    def doRead(self):
        raise Exception('Victim.doRead should never be called')

    def connectionLost(self, reason):
        """
                When the problem is detected, the reactor should disconnect this
                file descriptor.  When that happens, stop the reactor so the
                test ends.
                """
        reactor.stop()
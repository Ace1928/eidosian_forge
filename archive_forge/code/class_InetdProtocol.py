import os
from twisted.internet import fdesc, process, reactor
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.protocols import wire
class InetdProtocol(Protocol):
    """Forks a child process on connectionMade, passing the socket as fd 0."""

    def connectionMade(self):
        sockFD = self.transport.fileno()
        childFDs = {0: sockFD, 1: sockFD}
        if self.factory.stderrFile:
            childFDs[2] = self.factory.stderrFile.fileno()
        fdesc.setBlocking(sockFD)
        if 2 in childFDs:
            fdesc.setBlocking(childFDs[2])
        service = self.factory.service
        uid = service.user
        gid = service.group
        if uid == os.getuid():
            uid = None
        if gid == os.getgid():
            gid = None
        process.Process(None, service.program, service.programArgs, os.environ, None, None, uid, gid, childFDs)
        reactor.removeReader(self.transport)
        reactor.removeWriter(self.transport)
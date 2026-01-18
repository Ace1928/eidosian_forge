import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
@implementer(ISession)
class RekeyAvatar(ConchUser):
    """
    This avatar implements a shell which sends 60 numbered lines to whatever
    connects to it, then closes the session with a 0 exit status.

    60 lines is selected as being enough to send more than 2kB of traffic, the
    amount the client is configured to initiate a rekey after.
    """

    def __init__(self):
        ConchUser.__init__(self)
        self.channelLookup[b'session'] = SSHSession

    def openShell(self, transport):
        """
        Write 60 lines of data to the transport, then exit.
        """
        proto = protocol.Protocol()
        proto.makeConnection(transport)
        transport.makeConnection(wrapProtocol(proto))

        def write(counter):
            i = next(counter)
            if i == 60:
                call.stop()
                transport.session.conn.sendRequest(transport.session, b'exit-status', b'\x00\x00\x00\x00')
                transport.loseConnection()
            else:
                line = 'line #%02d\n' % (i,)
                line = line.encode('utf-8')
                transport.write(line)
        call = LoopingCall(write, count())
        call.start(0.01)

    def closed(self):
        """
        Ignore the close of the session.
        """

    def eofReceived(self):
        pass

    def execCommand(self, proto, command):
        pass

    def getPty(self, term, windowSize, modes):
        pass

    def windowChanged(self, newWindowSize):
        pass
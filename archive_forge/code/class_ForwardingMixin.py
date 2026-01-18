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
class ForwardingMixin(ConchServerSetupMixin):
    """
    Template class for tests of the Conch server's ability to forward arbitrary
    protocols over SSH.

    These tests are integration tests, not unit tests. They launch a Conch
    server, a custom TCP server (just an L{EchoProtocol}) and then call
    L{execute}.

    L{execute} is implemented by subclasses of L{ForwardingMixin}. It should
    cause an SSH client to connect to the Conch server, asking it to forward
    data to the custom TCP server.
    """

    def test_exec(self):
        """
        Test that we can use whatever client to send the command "echo goodbye"
        to the Conch server. Make sure we receive "goodbye" back from the
        server.
        """
        d = self.execute('echo goodbye', ConchTestOpenSSHProcess())
        return d.addCallback(self.assertEqual, b'goodbye\n')

    def test_localToRemoteForwarding(self):
        """
        Test that we can use whatever client to forward a local port to a
        specified port on the server.
        """
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -L%i:127.0.0.1:%i' % (localPort, self.echoPort))
        d.addCallback(self.assertEqual, b'test\n')
        return d

    def test_remoteToLocalForwarding(self):
        """
        Test that we can use whatever client to forward a port from the server
        to a port locally.
        """
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -R %i:127.0.0.1:%i' % (localPort, self.echoPort))
        d.addCallback(self.assertEqual, b'test\n')
        return d
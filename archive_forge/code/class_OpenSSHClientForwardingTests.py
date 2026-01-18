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
class OpenSSHClientForwardingTests(ForwardingMixin, OpenSSHClientMixin, TestCase):
    """
    Connection forwarding tests run against the OpenSSL command line client.
    """

    @skipIf(not HAS_IPV6, 'Requires IPv6 support')
    def test_localToRemoteForwardingV6(self):
        """
        Forwarding of arbitrary IPv6 TCP connections via SSH.
        """
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -L%i:[::1]:%i' % (localPort, self.echoPortV6))
        d.addCallback(self.assertEqual, b'test\n')
        return d
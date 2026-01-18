import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class StubAvatar:
    """
    A stub class representing the avatar representing the authenticated user.
    It implements the I{ISession} interface.
    """

    def lookupSubsystem(self, name, data):
        """
        If the user requests the TestSubsystem subsystem, connect them to a
        MockProtocol.  If they request neither, then None is returned which is
        interpreted by SSHSession as a failure.
        """
        if name == b'TestSubsystem':
            self.subsystem = MockProtocol()
            self.subsystem.packetData = data
            return self.subsystem
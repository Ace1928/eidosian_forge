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
class SubsystemOnlyAvatar:
    """
    A stub class representing an avatar that is only useful for
    getting a subsystem.
    """

    def lookupSubsystem(self, name, data):
        """
        If the other side requests the 'subsystem' subsystem, allow it by
        returning a MockProtocol to implement it. Otherwise raise an assertion.
        """
        assert name == b'subsystem'
        return MockProtocol()
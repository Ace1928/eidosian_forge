import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class MockService(service.SSHService):
    """
    A mocked-up service, based on twisted.conch.ssh.service.SSHService.

    @ivar started: True if this service has been started.
    @ivar stopped: True if this service has been stopped.
    """
    name = b'MockService'
    started = False
    stopped = False
    protocolMessages = {255: 'MSG_TEST', 71: 'MSG_fiction'}

    def logPrefix(self):
        return 'MockService'

    def serviceStarted(self):
        """
        Record that the service was started.
        """
        self.started = True

    def serviceStopped(self):
        """
        Record that the service was stopped.
        """
        self.stopped = True

    def ssh_TEST(self, packet):
        """
        A message that this service responds to.
        """
        self.transport.sendPacket(255, packet)
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
class ServerSSHTransportBaseCase(ServerAndClientSSHTransportBaseCase):
    """
    Base case for SSHServerTransport tests.
    """
    klass: Optional[Type[transport.SSHTransportBase]] = transport.SSHServerTransport

    def setUp(self):
        TransportTestCase.setUp(self)
        self.proto.factory = MockFactory()
        self.proto.factory.startFactory()

    def tearDown(self):
        TransportTestCase.tearDown(self)
        self.proto.factory.stopFactory()
        del self.proto.factory
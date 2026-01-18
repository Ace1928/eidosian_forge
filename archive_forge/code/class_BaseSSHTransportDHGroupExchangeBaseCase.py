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
class BaseSSHTransportDHGroupExchangeBaseCase(BaseSSHTransportBaseCase):
    """
    Diffie-Hellman group exchange tests for TransportBase.
    """

    def test_getKey(self):
        """
        Test that _getKey generates the correct keys.
        """
        self.proto.kexAlg = self.kexAlgorithm
        self.proto.sessionID = b'EF'
        k1 = self.hashProcessor(b'AB' + b'CD' + b'K' + self.proto.sessionID).digest()
        k2 = self.hashProcessor(b'ABCD' + k1).digest()
        k3 = self.hashProcessor(b'ABCD' + k1 + k2).digest()
        k4 = self.hashProcessor(b'ABCD' + k1 + k2 + k3).digest()
        self.assertEqual(self.proto._getKey(b'K', b'AB', b'CD'), k1 + k2 + k3 + k4)
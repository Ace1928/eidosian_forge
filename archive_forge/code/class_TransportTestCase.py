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
class TransportTestCase(TestCase):
    """
    Base class for transport test cases.
    """
    klass: Optional[Type[transport.SSHTransportBase]] = None
    if dependencySkip:
        skip = dependencySkip

    def setUp(self):
        self.transport = proto_helpers.StringTransport()
        self.proto = self.klass()
        self.packets = []

        def secureRandom(len):
            """
            Return a consistent entropy value
            """
            return b'\x99' * len
        self.patch(randbytes, 'secureRandom', secureRandom)
        self.proto._startEphemeralDH = types.MethodType(generatePredictableKey, self.proto)

        def stubSendPacket(messageType, payload):
            self.packets.append((messageType, payload))
        self.proto.makeConnection(self.transport)
        self.proto.sendPacket = stubSendPacket

    def finishKeyExchange(self, proto):
        """
        Deliver enough additional messages to C{proto} so that the key exchange
        which is started in L{SSHTransportBase.connectionMade} completes and
        non-key exchange messages can be sent and received.
        """
        proto.dataReceived(b'SSH-2.0-BogoClient-1.2i\r\n')
        proto.dispatchMessage(transport.MSG_KEXINIT, self._A_KEXINIT_MESSAGE)
        proto._keySetup(b'foo', b'bar')
        proto._keyExchangeState = proto._KEY_EXCHANGE_NONE

    def simulateKeyExchange(self, sharedSecret, exchangeHash):
        """
        Finish a key exchange by calling C{_keySetup} with the given arguments.
        Also do extra whitebox stuff to satisfy that method's assumption that
        some kind of key exchange has actually taken place.
        """
        self.proto._keyExchangeState = self.proto._KEY_EXCHANGE_REQUESTED
        self.proto._blockedByKeyExchange = []
        self.proto._keySetup(sharedSecret, exchangeHash)
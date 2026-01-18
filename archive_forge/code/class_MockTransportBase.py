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
class MockTransportBase(transport.SSHTransportBase):
    """
    A base class for the client and server protocols.  Stores the messages
    it receives instead of ignoring them.

    @ivar errors: a list of tuples: (reasonCode, description)
    @ivar unimplementeds: a list of integers: sequence number
    @ivar debugs: a list of tuples: (alwaysDisplay, message, lang)
    @ivar ignoreds: a list of strings: ignored data
    """

    def connectionMade(self):
        """
        Set up instance variables.
        """
        transport.SSHTransportBase.connectionMade(self)
        self.errors = []
        self.unimplementeds = []
        self.debugs = []
        self.ignoreds = []
        self.gotUnsupportedVersion = None

    def _unsupportedVersionReceived(self, remoteVersion):
        """
        Intercept unsupported version call.

        @type remoteVersion: L{str}
        """
        self.gotUnsupportedVersion = remoteVersion
        return transport.SSHTransportBase._unsupportedVersionReceived(self, remoteVersion)

    def receiveError(self, reasonCode, description):
        """
        Store any errors received.

        @type reasonCode: L{int}
        @type description: L{str}
        """
        self.errors.append((reasonCode, description))

    def receiveUnimplemented(self, seqnum):
        """
        Store any unimplemented packet messages.

        @type seqnum: L{int}
        """
        self.unimplementeds.append(seqnum)

    def receiveDebug(self, alwaysDisplay, message, lang):
        """
        Store any debug messages.

        @type alwaysDisplay: L{bool}
        @type message: L{str}
        @type lang: L{str}
        """
        self.debugs.append((alwaysDisplay, message, lang))

    def ssh_IGNORE(self, packet):
        """
        Store any ignored data.

        @type packet: L{str}
        """
        self.ignoreds.append(packet)
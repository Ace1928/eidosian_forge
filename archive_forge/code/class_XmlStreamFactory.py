from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
class XmlStreamFactory(xmlstream.XmlStreamFactory):
    """
    Factory for Jabber XmlStream objects as a reconnecting client.

    Note that this differs from L{xmlstream.XmlStreamFactory} in that
    it generates Jabber specific L{XmlStream} instances that have
    authenticators.
    """
    protocol = XmlStream

    def __init__(self, authenticator):
        xmlstream.XmlStreamFactory.__init__(self, authenticator)
        self.authenticator = authenticator
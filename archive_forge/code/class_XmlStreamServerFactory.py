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
class XmlStreamServerFactory(xmlstream.BootstrapMixin, protocol.ServerFactory):
    """
    Factory for Jabber XmlStream objects as a server.

    @since: 8.2.
    @ivar authenticatorFactory: Factory callable that takes no arguments, to
                                create a fresh authenticator to be associated
                                with the XmlStream.
    """
    protocol = XmlStream

    def __init__(self, authenticatorFactory):
        xmlstream.BootstrapMixin.__init__(self)
        self.authenticatorFactory = authenticatorFactory

    def buildProtocol(self, addr):
        """
        Create an instance of XmlStream.

        A new authenticator instance will be created and passed to the new
        XmlStream. Registered bootstrap event observers are installed as well.
        """
        authenticator = self.authenticatorFactory()
        xs = self.protocol(authenticator)
        xs.factory = self
        self.installBootstraps(xs)
        return xs
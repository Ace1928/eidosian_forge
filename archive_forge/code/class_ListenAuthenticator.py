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
class ListenAuthenticator(Authenticator):
    """
    Authenticator for receiving entities.
    """
    namespace: Optional[str] = None

    def associateWithStream(self, xmlstream):
        """
        Called by the XmlStreamFactory when a connection has been made.

        Extend L{Authenticator.associateWithStream} to set the L{XmlStream}
        to be non-initiating.
        """
        Authenticator.associateWithStream(self, xmlstream)
        self.xmlstream.initiating = False

    def streamStarted(self, rootElement):
        """
        Called by the XmlStream when the stream has started.

        This extends L{Authenticator.streamStarted} to extract further
        information from the stream headers from C{rootElement}.
        """
        Authenticator.streamStarted(self, rootElement)
        self.xmlstream.namespace = rootElement.defaultUri
        if rootElement.hasAttribute('to'):
            self.xmlstream.thisEntity = jid.internJID(rootElement['to'])
        self.xmlstream.prefixes = {}
        for prefix, uri in rootElement.localPrefixes.items():
            self.xmlstream.prefixes[uri] = prefix
        self.xmlstream.sid = hexlify(randbytes.secureRandom(8)).decode('ascii')
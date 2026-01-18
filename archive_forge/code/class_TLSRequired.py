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
class TLSRequired(TLSError):
    """
    Exception indicating required TLS negotiation.

    This exception is raised when the receiving entity requires TLS
    negotiation and the initiating does not desire to negotiate TLS.
    """
from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class INegotiated(ISSLTransport):
    """
    A TLS based transport that supports using ALPN/NPN to negotiate the
    protocol to be used inside the encrypted tunnel.
    """
    negotiatedProtocol = Attribute('\n        The protocol selected to be spoken using ALPN/NPN. The result from ALPN\n        is preferred to the result from NPN if both were used. If the remote\n        peer does not support ALPN or NPN, or neither NPN or ALPN are available\n        on this machine, will be L{None}. Otherwise, will be the name of the\n        selected protocol as C{bytes}. Note that until the handshake has\n        completed this property may incorrectly return L{None}: wait until data\n        has been received before trusting it (see\n        https://twistedmatrix.com/trac/ticket/6024).\n        ')
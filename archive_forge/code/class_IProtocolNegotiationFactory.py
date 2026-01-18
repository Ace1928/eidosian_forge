from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IProtocolNegotiationFactory(Interface):
    """
    A provider of L{IProtocolNegotiationFactory} can provide information about
    the various protocols that the factory can create implementations of. This
    can be used, for example, to provide protocol names for Next Protocol
    Negotiation and Application Layer Protocol Negotiation.

    @see: L{twisted.internet.ssl}
    """

    def acceptableProtocols() -> List[bytes]:
        """
        Returns a list of protocols that can be spoken by the connection
        factory in the form of ALPN tokens, as laid out in the IANA registry
        for ALPN tokens.

        @return: a list of ALPN tokens in order of preference.
        """
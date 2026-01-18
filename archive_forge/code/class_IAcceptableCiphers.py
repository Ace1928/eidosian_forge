from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IAcceptableCiphers(Interface):
    """
    A list of acceptable ciphers for a TLS context.
    """

    def selectCiphers(availableCiphers: Tuple[ICipher]) -> Tuple[ICipher]:
        """
        Choose which ciphers to allow to be negotiated on a TLS connection.

        @param availableCiphers: A L{tuple} of L{ICipher} which gives the names
            of all ciphers supported by the TLS implementation in use.

        @return: A L{tuple} of L{ICipher} which represents the ciphers
            which may be negotiated on the TLS connection.  The result is
            ordered by preference with more preferred ciphers appearing
            earlier.
        """
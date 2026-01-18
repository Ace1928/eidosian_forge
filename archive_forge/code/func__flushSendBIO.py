from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
def _flushSendBIO(self):
    """
        Read any bytes out of the send BIO and write them to the underlying
        transport.
        """
    try:
        bytes = self._tlsConnection.bio_read(2 ** 15)
    except WantReadError:
        pass
    else:
        self.transport.write(bytes)
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
def _flushReceiveBIO(self):
    """
        Try to receive any application-level bytes which are now available
        because of a previous write into the receive BIO.  This will take
        care of delivering any application-level bytes which are received to
        the protocol, as well as handling of the various exceptions which
        can come from trying to get such bytes.
        """
    while not self._lostTLSConnection:
        try:
            bytes = self._tlsConnection.recv(2 ** 15)
        except WantReadError:
            break
        except ZeroReturnError:
            self._shutdownTLS()
            self._tlsShutdownFinished(None)
        except Error:
            failure = Failure()
            self._tlsShutdownFinished(failure)
        else:
            if not self._aborted:
                ProtocolWrapper.dataReceived(self, bytes)
    self._flushSendBIO()
from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
def _startTLS(self, certificate, verifyAuthorities):
    """
        Used by TLSBox to initiate the SSL handshake.

        @param certificate: a L{twisted.internet.ssl.PrivateCertificate} for
        use locally.

        @param verifyAuthorities: L{twisted.internet.ssl.Certificate} instances
        representing certificate authorities which will verify our peer.
        """
    self.hostCertificate = certificate
    self._justStartedTLS = True
    if verifyAuthorities is None:
        verifyAuthorities = ()
    self.transport.startTLS(certificate.options(*verifyAuthorities))
    stlsb = self._startingTLSBuffer
    if stlsb is not None:
        self._startingTLSBuffer = None
        for box in stlsb:
            self.sendBox(box)
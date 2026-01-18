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
@implementer(IBoxSender)
class BinaryBoxProtocol(StatefulStringProtocol, Int16StringReceiver, _DescriptorExchanger):
    """
    A protocol for receiving L{AmpBox}es - key/value pairs - via length-prefixed
    strings.  A box is composed of:

        - any number of key-value pairs, described by:
            - a 2-byte network-endian packed key length (of which the first
              byte must be null, and the second must be non-null: i.e. the
              value of the length must be 1-255)
            - a key, comprised of that many bytes
            - a 2-byte network-endian unsigned value length (up to the maximum
              of 65535)
            - a value, comprised of that many bytes
        - 2 null bytes

    In other words, an even number of strings prefixed with packed unsigned
    16-bit integers, and then a 0-length string to indicate the end of the box.

    This protocol also implements 2 extra private bits of functionality related
    to the byte boundaries between messages; it can start TLS between two given
    boxes or switch to an entirely different protocol.  However, due to some
    tricky elements of the implementation, the public interface to this
    functionality is L{ProtocolSwitchCommand} and L{StartTLS}.

    @ivar _keyLengthLimitExceeded: A flag which is only true when the
        connection is being closed because a key length prefix which was longer
        than allowed by the protocol was received.

    @ivar boxReceiver: an L{IBoxReceiver} provider, whose
        L{IBoxReceiver.ampBoxReceived} method will be invoked for each
        L{AmpBox} that is received.
    """
    _justStartedTLS = False
    _startingTLSBuffer = None
    _locked = False
    _currentKey = None
    _currentBox = None
    _keyLengthLimitExceeded = False
    hostCertificate = None
    noPeerCertificate = False
    innerProtocol: Optional[Protocol] = None
    innerProtocolClientFactory = None

    def __init__(self, boxReceiver):
        _DescriptorExchanger.__init__(self)
        self.boxReceiver = boxReceiver

    def _switchTo(self, newProto, clientFactory=None):
        """
        Switch this BinaryBoxProtocol's transport to a new protocol.  You need
        to do this 'simultaneously' on both ends of a connection; the easiest
        way to do this is to use a subclass of ProtocolSwitchCommand.

        @param newProto: the new protocol instance to switch to.

        @param clientFactory: the ClientFactory to send the
            L{twisted.internet.protocol.ClientFactory.clientConnectionLost}
            notification to.
        """
        newProtoData = self.recvd
        self.recvd = ''
        self.innerProtocol = newProto
        self.innerProtocolClientFactory = clientFactory
        newProto.makeConnection(self.transport)
        if newProtoData:
            newProto.dataReceived(newProtoData)

    def sendBox(self, box):
        """
        Send a amp.Box to my peer.

        Note: transport.write is never called outside of this method.

        @param box: an AmpBox.

        @raise ProtocolSwitched: if the protocol has previously been switched.

        @raise ConnectionLost: if the connection has previously been lost.
        """
        if self._locked:
            raise ProtocolSwitched('This connection has switched: no AMP traffic allowed.')
        if self.transport is None:
            raise ConnectionLost()
        if self._startingTLSBuffer is not None:
            self._startingTLSBuffer.append(box)
        else:
            self.transport.write(box.serialize())

    def makeConnection(self, transport):
        """
        Notify L{boxReceiver} that it is about to receive boxes from this
        protocol by invoking L{IBoxReceiver.startReceivingBoxes}.
        """
        self.transport = transport
        self.boxReceiver.startReceivingBoxes(self)
        self.connectionMade()

    def dataReceived(self, data):
        """
        Either parse incoming data as L{AmpBox}es or relay it to our nested
        protocol.
        """
        if self._justStartedTLS:
            self._justStartedTLS = False
        if self.innerProtocol is not None:
            self.innerProtocol.dataReceived(data)
            return
        return Int16StringReceiver.dataReceived(self, data)

    def connectionLost(self, reason):
        """
        The connection was lost; notify any nested protocol.
        """
        if self.innerProtocol is not None:
            self.innerProtocol.connectionLost(reason)
            if self.innerProtocolClientFactory is not None:
                self.innerProtocolClientFactory.clientConnectionLost(None, reason)
        if self._keyLengthLimitExceeded:
            failReason = Failure(TooLong(True, False, None, None))
        elif reason.check(ConnectionClosed) and self._justStartedTLS:
            failReason = PeerVerifyError('Peer rejected our certificate for an unknown reason.')
        else:
            failReason = reason
        self.boxReceiver.stopReceivingBoxes(failReason)
    _MAX_KEY_LENGTH = 255
    _MAX_VALUE_LENGTH = 65535
    MAX_LENGTH = _MAX_KEY_LENGTH

    def proto_init(self, string):
        """
        String received in the 'init' state.
        """
        self._currentBox = AmpBox()
        return self.proto_key(string)

    def proto_key(self, string):
        """
        String received in the 'key' state.  If the key is empty, a complete
        box has been received.
        """
        if string:
            self._currentKey = string
            self.MAX_LENGTH = self._MAX_VALUE_LENGTH
            return 'value'
        else:
            self.boxReceiver.ampBoxReceived(self._currentBox)
            self._currentBox = None
            return 'init'

    def proto_value(self, string):
        """
        String received in the 'value' state.
        """
        self._currentBox[self._currentKey] = string
        self._currentKey = None
        self.MAX_LENGTH = self._MAX_KEY_LENGTH
        return 'key'

    def lengthLimitExceeded(self, length):
        """
        The key length limit was exceeded.  Disconnect the transport and make
        sure a meaningful exception is reported.
        """
        self._keyLengthLimitExceeded = True
        self.transport.loseConnection()

    def _lockForSwitch(self):
        """
        Lock this binary protocol so that no further boxes may be sent.  This
        is used when sending a request to switch underlying protocols.  You
        probably want to subclass ProtocolSwitchCommand rather than calling
        this directly.
        """
        self._locked = True

    def _unlockFromSwitch(self):
        """
        Unlock this locked binary protocol so that further boxes may be sent
        again.  This is used after an attempt to switch protocols has failed
        for some reason.
        """
        if self.innerProtocol is not None:
            raise ProtocolSwitched('Protocol already switched.  Cannot unlock.')
        self._locked = False

    def _prepareTLS(self, certificate, verifyAuthorities):
        """
        Used by StartTLSCommand to put us into the state where we don't
        actually send things that get sent, instead we buffer them.  see
        L{_sendBoxCommand}.
        """
        self._startingTLSBuffer = []
        if self.hostCertificate is not None:
            raise OnlyOneTLS('Previously authenticated connection between %s and %s is trying to re-establish as %s' % (self.hostCertificate, self.peerCertificate, (certificate, verifyAuthorities)))

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

    @property
    def peerCertificate(self):
        if self.noPeerCertificate:
            return None
        return Certificate.peerFromTransport(self.transport)

    def unhandledError(self, failure):
        """
        The buck stops here.  This error was completely unhandled, time to
        terminate the connection.
        """
        log.err(failure, 'Amp server or network failure unhandled by client application.  Dropping connection!  To avoid, add errbacks to ALL remote commands!')
        if self.transport is not None:
            self.transport.loseConnection()

    def _defaultStartTLSResponder(self):
        """
        The default TLS responder doesn't specify any certificate or anything.

        From a security perspective, it's little better than a plain-text
        connection - but it is still a *bit* better, so it's included for
        convenience.

        You probably want to override this by providing your own StartTLS.responder.
        """
        return {}
    StartTLS.responder(_defaultStartTLSResponder)
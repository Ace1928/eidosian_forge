import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
class DTPFactory(protocol.ClientFactory):
    """
    Client factory for I{data transfer process} protocols.

    @ivar peerCheck: perform checks to make sure the ftp-pi's peer is the same
        as the dtp's
    @ivar pi: a reference to this factory's protocol interpreter

    @ivar _state: Indicates the current state of the DTPFactory.  Initially,
        this is L{_IN_PROGRESS}.  If the connection fails or times out, it is
        L{_FAILED}.  If the connection succeeds before the timeout, it is
        L{_FINISHED}.

    @cvar _IN_PROGRESS: Token to signal that connection is active.
    @type _IN_PROGRESS: L{object}.

    @cvar _FAILED: Token to signal that connection has failed.
    @type _FAILED: L{object}.

    @cvar _FINISHED: Token to signal that connection was successfully closed.
    @type _FINISHED: L{object}.
    """
    _IN_PROGRESS = object()
    _FAILED = object()
    _FINISHED = object()
    _state = _IN_PROGRESS
    peerCheck = False

    def __init__(self, pi, peerHost=None, reactor=None):
        """
        Constructor

        @param pi: this factory's protocol interpreter
        @param peerHost: if peerCheck is True, this is the tuple that the
            generated instance will use to perform security checks
        """
        self.pi = pi
        self.peerHost = peerHost
        self.deferred = defer.Deferred()
        self.delayedCall = None
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor

    def buildProtocol(self, addr):
        log.msg('DTPFactory.buildProtocol', debug=True)
        if self._state is not self._IN_PROGRESS:
            return None
        self._state = self._FINISHED
        self.cancelTimeout()
        p = DTP()
        p.factory = self
        p.pi = self.pi
        self.pi.dtpInstance = p
        return p

    def stopFactory(self):
        log.msg('dtpFactory.stopFactory', debug=True)
        self.cancelTimeout()

    def timeoutFactory(self):
        log.msg('timed out waiting for DTP connection')
        if self._state is not self._IN_PROGRESS:
            return
        self._state = self._FAILED
        d = self.deferred
        self.deferred = None
        d.errback(PortConnectionError(defer.TimeoutError('DTPFactory timeout')))

    def cancelTimeout(self):
        if self.delayedCall is not None and self.delayedCall.active():
            log.msg('cancelling DTP timeout', debug=True)
            self.delayedCall.cancel()

    def setTimeout(self, seconds):
        log.msg('DTPFactory.setTimeout set to %s seconds' % seconds)
        self.delayedCall = self._reactor.callLater(seconds, self.timeoutFactory)

    def clientConnectionFailed(self, connector, reason):
        if self._state is not self._IN_PROGRESS:
            return
        self._state = self._FAILED
        d = self.deferred
        self.deferred = None
        d.errback(PortConnectionError(reason))
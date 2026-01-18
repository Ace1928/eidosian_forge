import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
class ReconnectingClientFactory(ClientFactory):
    """
    Factory which auto-reconnects clients with an exponential back-off.

    Note that clients should call my resetDelay method after they have
    connected successfully.

    @ivar maxDelay: Maximum number of seconds between connection attempts.
    @ivar initialDelay: Delay for the first reconnection attempt.
    @ivar factor: A multiplicitive factor by which the delay grows
    @ivar jitter: Percentage of randomness to introduce into the delay length
        to prevent stampeding.
    @ivar clock: The clock used to schedule reconnection. It's mainly useful to
        be parametrized in tests. If the factory is serialized, this attribute
        will not be serialized, and the default value (the reactor) will be
        restored when deserialized.
    @type clock: L{IReactorTime}
    @ivar maxRetries: Maximum number of consecutive unsuccessful connection
        attempts, after which no further connection attempts will be made. If
        this is not explicitly set, no maximum is applied.
    """
    maxDelay = 3600
    initialDelay = 1.0
    factor = 2.718281828459045
    jitter = 0.119626565582
    delay = initialDelay
    retries = 0
    maxRetries = None
    _callID = None
    connector = None
    clock = None
    continueTrying = 1

    def clientConnectionFailed(self, connector, reason):
        if self.continueTrying:
            self.connector = connector
            self.retry()

    def clientConnectionLost(self, connector, unused_reason):
        if self.continueTrying:
            self.connector = connector
            self.retry()

    def retry(self, connector=None):
        """
        Have this connector connect again, after a suitable delay.
        """
        if not self.continueTrying:
            if self.noisy:
                log.msg(f'Abandoning {connector} on explicit request')
            return
        if connector is None:
            if self.connector is None:
                raise ValueError('no connector to retry')
            else:
                connector = self.connector
        self.retries += 1
        if self.maxRetries is not None and self.retries > self.maxRetries:
            if self.noisy:
                log.msg('Abandoning %s after %d retries.' % (connector, self.retries))
            return
        self.delay = min(self.delay * self.factor, self.maxDelay)
        if self.jitter:
            self.delay = random.normalvariate(self.delay, self.delay * self.jitter)
        if self.noisy:
            log.msg('%s will retry in %d seconds' % (connector, self.delay))

        def reconnector():
            self._callID = None
            connector.connect()
        if self.clock is None:
            from twisted.internet import reactor
            self.clock = reactor
        self._callID = self.clock.callLater(self.delay, reconnector)

    def stopTrying(self):
        """
        Put a stop to any attempt to reconnect in progress.
        """
        if self._callID:
            self._callID.cancel()
            self._callID = None
        self.continueTrying = 0
        if self.connector:
            try:
                self.connector.stopConnecting()
            except error.NotConnectingError:
                pass

    def resetDelay(self):
        """
        Call this method after a successful connection: it resets the delay and
        the retry counter.
        """
        self.delay = self.initialDelay
        self.retries = 0
        self._callID = None
        self.continueTrying = 1

    def __getstate__(self):
        """
        Remove all of the state which is mutated by connection attempts and
        failures, returning just the state which describes how reconnections
        should be attempted.  This will make the unserialized instance
        behave just as this one did when it was first instantiated.
        """
        state = self.__dict__.copy()
        for key in ['connector', 'retries', 'delay', 'continueTrying', '_callID', 'clock']:
            if key in state:
                del state[key]
        return state
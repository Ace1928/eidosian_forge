from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class StreamServerEndpointService(service.Service):
    """
    A L{StreamServerEndpointService} is an L{IService} which runs a server on a
    listening port described by an L{IStreamServerEndpoint
    <twisted.internet.interfaces.IStreamServerEndpoint>}.

    @ivar factory: A server factory which will be used to listen on the
        endpoint.

    @ivar endpoint: An L{IStreamServerEndpoint
        <twisted.internet.interfaces.IStreamServerEndpoint>} provider
        which will be used to listen when the service starts.

    @ivar _waitingForPort: a Deferred, if C{listen} has yet been invoked on the
        endpoint, otherwise None.

    @ivar _raiseSynchronously: Defines error-handling behavior for the case
        where C{listen(...)} raises an exception before C{startService} or
        C{privilegedStartService} have completed.

    @type _raiseSynchronously: C{bool}

    @since: 10.2
    """
    _raiseSynchronously = False

    def __init__(self, endpoint, factory):
        self.endpoint = endpoint
        self.factory = factory
        self._waitingForPort = None

    def privilegedStartService(self):
        """
        Start listening on the endpoint.
        """
        service.Service.privilegedStartService(self)
        self._waitingForPort = self.endpoint.listen(self.factory)
        raisedNow = []

        def handleIt(err):
            if self._raiseSynchronously:
                raisedNow.append(err)
            elif not err.check(CancelledError):
                log.err(err)
        self._waitingForPort.addErrback(handleIt)
        if raisedNow:
            raisedNow[0].raiseException()
        self._raiseSynchronously = False

    def startService(self):
        """
        Start listening on the endpoint, unless L{privilegedStartService} got
        around to it already.
        """
        service.Service.startService(self)
        if self._waitingForPort is None:
            self.privilegedStartService()

    def stopService(self):
        """
        Stop listening on the port if it is already listening, otherwise,
        cancel the attempt to listen.

        @return: a L{Deferred<twisted.internet.defer.Deferred>} which fires
            with L{None} when the port has stopped listening.
        """
        self._waitingForPort.cancel()

        def stopIt(port):
            if port is not None:
                return port.stopListening()
        d = self._waitingForPort.addCallback(stopIt)

        def stop(passthrough):
            self.running = False
            return passthrough
        d.addBoth(stop)
        return d
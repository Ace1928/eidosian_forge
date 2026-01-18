from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class _AbstractServer(_VolatileDataService):
    """
    @cvar volatile: list of attribute to remove from pickling.
    @type volatile: C{list}

    @ivar method: the type of method to call on the reactor, one of B{TCP},
        B{UDP}, B{SSL} or B{UNIX}.
    @type method: C{str}

    @ivar reactor: the current running reactor.
    @type reactor: a provider of C{IReactorTCP}, C{IReactorUDP},
        C{IReactorSSL} or C{IReactorUnix}.

    @ivar _port: instance of port set when the service is started.
    @type _port: a provider of L{twisted.internet.interfaces.IListeningPort}.
    """
    volatile = ['_port']
    method: str = ''
    reactor = None
    _port = None

    def __init__(self, *args, **kwargs):
        self.args = args
        if 'reactor' in kwargs:
            self.reactor = kwargs.pop('reactor')
        self.kwargs = kwargs

    def privilegedStartService(self):
        service.Service.privilegedStartService(self)
        self._port = self._getPort()

    def startService(self):
        service.Service.startService(self)
        if self._port is None:
            self._port = self._getPort()

    def stopService(self):
        service.Service.stopService(self)
        if self._port is not None:
            d = self._port.stopListening()
            del self._port
            return d

    def _getPort(self):
        """
        Wrapper around the appropriate listen method of the reactor.

        @return: the port object returned by the listen method.
        @rtype: an object providing
            L{twisted.internet.interfaces.IListeningPort}.
        """
        return getattr(_maybeGlobalReactor(self.reactor), 'listen{}'.format(self.method))(*self.args, **self.kwargs)